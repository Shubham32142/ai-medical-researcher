from __future__ import annotations

import asyncio
import re
import time
from collections import Counter
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv(".env.local", override=True)
load_dotenv()

from app.retrievers.clinicaltrials import fetch_clinical_trials
from app.retrievers.openalex import fetch_openalex
from app.retrievers.pubmed import fetch_pubmed
from app.services.llm import generate_with_optional_llm

app = FastAPI(title="CuraLink AI", version="0.1.0")
CACHE_TTL_SECONDS = 1800
research_cache: dict[str, tuple[float, dict[str, Any]]] = {}
STOPWORDS = {
    "latest",
    "what",
    "which",
    "show",
    "find",
    "recent",
    "about",
    "directions",
    "options",
    "india",
    "mumbai",
    "delhi",
    "please",
}
TREATMENT_TERMS = {
    "treatment",
    "therapy",
    "therapies",
    "immunotherapy",
    "chemotherapy",
    "radiation",
    "surgery",
    "targeted",
    "management",
    "trial",
    "trials",
}
TRIAL_TERMS = {
    "trial",
    "trials",
    "recruiting",
    "recruitment",
    "study",
    "studies",
    "eligibility",
}
SIDE_EFFECT_TERMS = {
    "side effect",
    "side effects",
    "interaction",
    "interactions",
    "adverse",
    "toxicity",
    "toxicities",
    "safety",
    "risk",
    "risks",
    "harm",
    "harms",
    "pneumonitis",
}
NUTRITION_TERMS = {
    "diet",
    "food",
    "foods",
    "fruit",
    "fruits",
    "nutrition",
    "meal",
    "glycemic",
    "fiber",
}
GENERIC_PENALTY_TERMS = {
    "global burden",
    "incidence",
    "mortality",
    "years of life lost",
    "disability-adjusted",
    "epidemiology",
    "prevalence estimates",
    "diabetes atlas",
    "international diabetes federation",
}
COMMON_CONDITIONS = {
    "diabetes",
    "cancer",
    "asthma",
    "arthritis",
    "hypertension",
    "obesity",
    "depression",
    "anxiety",
    "migraine",
    "thyroid",
    "eczema",
    "psoriasis",
    "covid",
    "copd",
    "stroke",
    "kidney",
    "liver",
    "heart",
    "lung",
    "breast",
}
FOLLOWUP_CUES = {
    "what about",
    "how about",
    "tell me more",
    "what else",
    "side effects",
    "risks",
    "survival",
    "prognosis",
    "cost",
}


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    sessionId: str
    disease: str
    location: str = ""
    history: list[ChatTurn] = Field(default_factory=list)
    message: str


class ResearchRequest(BaseModel):
    query: str
    location: str = ""


def tokenize(text: str) -> list[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return [part for part in cleaned.split() if len(part) > 2]


def extract_keywords(text: str) -> list[str]:
    return [token for token in tokenize(text) if token not in STOPWORDS]


def has_treatment_intent(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in TREATMENT_TERMS)


def is_trial_query(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in TRIAL_TERMS)


def is_side_effect_query(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in SIDE_EFFECT_TERMS)


def has_nutrition_intent(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in NUTRITION_TERMS)


def is_prescription_request(text: str) -> bool:
    lowered = text.lower()
    markers = [
        "dosage",
        "dose",
        "how much",
        "mg",
        "prescribe",
        "prescription",
        "should i take",
        "how many times",
    ]
    return any(marker in lowered for marker in markers)


def extract_condition_phrase(text: str) -> str | None:
    lowered = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    patterns = [
        r"\b(type\s*[12]\s+diabetes|gestational diabetes|prediabetes|diabetes)\b",
        r"\b([a-z]+\s+){0,2}cancer\b",
        r"\b(rheumatoid arthritis|arthritis|asthma|hypertension|depression|anxiety|migraine|eczema|psoriasis|thyroid disease|copd|covid)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return match.group(0).strip()
    return None


def has_condition_signal(text: str) -> bool:
    tokens = extract_keywords(text)
    medical_suffixes = ("itis", "osis", "emia", "oma", "pathy", "plasia", "penia")
    return bool(extract_condition_phrase(text)) or any(token in COMMON_CONDITIONS or token.endswith(medical_suffixes) for token in tokens)


def is_vague_followup(text: str) -> bool:
    lowered = text.lower().strip()
    if any(cue in lowered for cue in FOLLOWUP_CUES):
        return True
    tokens = extract_keywords(text)
    return len(tokens) <= 2 and not has_condition_signal(text)


def resolve_focus_topic(disease: str, message: str) -> str:
    disease_terms = set(extract_keywords(disease))
    message_terms = set(extract_keywords(message))
    condition_phrase = extract_condition_phrase(message)

    if disease_terms & message_terms:
        return disease
    if is_vague_followup(message):
        return disease
    if condition_phrase:
        return condition_phrase
    if has_condition_signal(message):
        return message
    return disease


def is_explicit_topic_switch(session_disease: str, message: str, focus_topic: str) -> bool:
    mentioned_condition = extract_condition_phrase(message)
    if not mentioned_condition:
        return False

    session_terms = set(extract_keywords(session_disease))
    focus_terms = set(extract_keywords(focus_topic))
    return bool(focus_terms) and session_terms.isdisjoint(focus_terms)


def summarize_snippet(text: str, limit: int = 180) -> str:
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return "No abstract summary was available."

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    if sentences and len(sentences[0]) <= limit:
        return sentences[0]

    trimmed = cleaned[:limit].rstrip()
    if " " in trimmed:
        trimmed = trimmed.rsplit(" ", 1)[0]
    return trimmed + "…"


def llm_answer_looks_relevant(answer: str, query: str) -> bool:
    cleaned = " ".join(answer.split()).strip()
    if len(cleaned) < 80:
        return False

    answer_terms = set(tokenize(cleaned))
    query_terms = set(extract_keywords(query))
    if query_terms and answer_terms.intersection(query_terms):
        return True

    return any(
        phrase in cleaned.lower()
        for phrase in ["trial", "side effect", "interaction", "evidence", "treatment", "therapy", "doctor"]
    )


def sanitize_llm_text(answer: str) -> str:
    cleaned = re.sub(r"\[(?:pmid:?\s*)?\d+\]", "", answer, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[W\d+\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def evidence_is_specific_enough(publications: list[dict[str, Any]], disease: str, message: str) -> bool:
    if not publications:
        return False

    if has_nutrition_intent(message):
        disease_terms = set(extract_keywords(disease))
        message_terms = set(extract_keywords(message))
        extra_terms = {
            term for term in message_terms if term not in disease_terms and term not in {"this", "disease"}
        }
        intent_terms = extra_terms or NUTRITION_TERMS

        for item in publications[:3]:
            text_terms = set(tokenize(f"{item.get('title', '')} {item.get('abstract', '')}"))
            if text_terms.intersection(intent_terms) and text_terms.intersection(NUTRITION_TERMS):
                return True
        return False

    return True


def build_query_variants(disease: str, message: str, location: str, history: str) -> list[str]:
    disease_phrase = f'"{disease}"' if " " in disease else disease
    message_terms = " ".join(extract_keywords(message)[:4])
    history_terms = " ".join(extract_keywords(history)[:3])
    is_nutrition_query = any(term in message.lower() for term in {"diet", "food", "foods", "fruit", "fruits", "eat", "nutrition"})

    if is_trial_query(message):
        candidates = [
            f"{disease_phrase} recruiting clinical trials {location}".strip(),
            f"{disease_phrase} interventional study recruiting".strip(),
            f"{disease_phrase} clinical trial eligibility therapy".strip(),
            f"{disease_phrase} {message_terms}".strip(),
        ]
    elif is_side_effect_query(message):
        candidates = [
            f"{disease_phrase} treatment adverse effects interactions review".strip(),
            f"{disease_phrase} immunotherapy chemotherapy toxicity review".strip(),
            f"{disease_phrase} targeted therapy adverse events".strip(),
            f"{disease_phrase} {message_terms}".strip(),
        ]
    elif has_treatment_intent(message):
        candidates = [
            f"{disease_phrase} {message_terms} treatment therapy".strip(),
            f"{disease_phrase} treatment therapy review".strip(),
            f"{disease_phrase} immunotherapy chemotherapy clinical trial".strip(),
            f"{disease_phrase} {history_terms}".strip(),
        ]
    else:
        candidates = [
            f"{disease_phrase} {message_terms} clinical evidence".strip(),
            f"{disease_phrase} {message_terms} systematic review".strip(),
            f"{disease_phrase} diet nutrition review".strip() if is_nutrition_query else f"{disease_phrase} clinical review".strip(),
            f"{disease_phrase} {history_terms}".strip(),
        ]

    seen: set[str] = set()
    variants: list[str] = []
    for item in candidates:
        normalized = " ".join(item.split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            variants.append(normalized)
    return variants[:4]


def location_matches(location: str, trial: dict[str, Any]) -> bool:
    if not location:
        return False
    haystack = " ".join(
        f"{place.get('city', '')} {place.get('country', '')}" for place in trial.get("locations", [])
    ).lower()
    return any(token in haystack for token in tokenize(location))


def score_item(query: str, item: dict[str, Any], disease: str, message: str) -> float:
    title = item.get("title", "")
    text = " ".join(
        [
            title,
            item.get("abstract", ""),
            item.get("journal", ""),
            " ".join(item.get("conditions", [])),
            " ".join(item.get("interventions", [])),
        ]
    )
    q_count = Counter(tokenize(query))
    d_count = Counter(tokenize(text))
    overlap = sum(min(q_count[key], d_count[key]) for key in q_count)

    disease_terms = extract_keywords(disease)
    title_terms = set(tokenize(title))
    body_terms = set(tokenize(text))
    disease_hits = sum(1 for term in disease_terms if term in body_terms)
    disease_title_hits = sum(1 for term in disease_terms if term in title_terms)
    disease_bonus = (disease_hits * 1.8) + (disease_title_hits * 2.4)

    intent_hits = sum(1 for term in TREATMENT_TERMS if term in body_terms)
    nutrition_hits = sum(1 for term in NUTRITION_TERMS if term in body_terms)
    intent_bonus = intent_hits * 1.2 if has_treatment_intent(message) else 0
    nutrition_bonus = nutrition_hits * 1.1 if has_nutrition_intent(message) else 0

    recency = max(0, int(item.get("year", 2024)) - 2017) / 10
    source_bonus = 0.55 if item.get("source") == "pubmed" else 0.15
    trial_bonus = 0.7 if item.get("nearUser") else 0
    abstract_bonus = 0.15 if item.get("abstract") else 0

    lowered_text = text.lower()
    generic_penalty = 0
    if any(term in lowered_text for term in GENERIC_PENALTY_TERMS) and not any(term in lowered_text for term in TREATMENT_TERMS):
        generic_penalty = 3.0

    return round((overlap * 1.2) + disease_bonus + intent_bonus + nutrition_bonus + recency + source_bonus + trial_bonus + abstract_bonus - generic_penalty, 3)


def filter_publications(publications: list[dict[str, Any]], disease: str, message: str) -> list[dict[str, Any]]:
    disease_terms = extract_keywords(disease)
    require_trial = is_trial_query(message)
    require_side_effects = is_side_effect_query(message)
    require_treatment = has_treatment_intent(message)
    require_nutrition = has_nutrition_intent(message)
    filtered: list[dict[str, Any]] = []

    for item in publications:
        if int(item.get("year", 2024)) < 2018:
            continue

        lowered_text = f"{item.get('title', '')} {item.get('abstract', '')}".lower()
        disease_hits = sum(1 for term in disease_terms if term in lowered_text)
        if disease_hits == 0:
            continue

        text_terms = set(tokenize(lowered_text))
        safety_terms = {"adverse", "toxicity", "toxicities", "interaction", "interactions", "safety", "pneumonitis", "complication", "complications"}

        if require_trial:
            if "trial" in lowered_text or "study" in lowered_text or text_terms.intersection({"recruiting", "interventional"}):
                filtered.append(item)
        elif require_side_effects:
            if text_terms.intersection(safety_terms) or "side effect" in lowered_text:
                filtered.append(item)
        elif require_treatment:
            if text_terms.intersection(TREATMENT_TERMS) or "review" in lowered_text:
                filtered.append(item)
        elif require_nutrition:
            if text_terms.intersection(NUTRITION_TERMS):
                filtered.append(item)
        else:
            filtered.append(item)

    if filtered:
        return filtered

    fallback = [item for item in publications if int(item.get("year", 2024)) >= 2018]
    return fallback or publications


def get_cached_result(cache_key: str) -> dict[str, Any] | None:
    entry = research_cache.get(cache_key)
    if not entry:
        return None
    created_at, value = entry
    if time.time() - created_at > CACHE_TTL_SECONDS:
        research_cache.pop(cache_key, None)
        return None
    return value


def set_cached_result(cache_key: str, value: dict[str, Any]) -> None:
    research_cache[cache_key] = (time.time(), value)


async def gather_research(query: str, disease: str, location: str, history: str = "") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    variants = build_query_variants(disease, query, location, history)
    cache_key = "|".join(variants) + f"|{location}"
    cached = get_cached_result(cache_key)
    if cached:
        return cached["publications"], cached["trials"]

    tasks = []
    for variant in variants:
        tasks.extend(
            [
                fetch_pubmed(variant, limit=10),
                fetch_openalex(variant, limit=8),
            ]
        )

    trial_query = f"{disease} treatment" if has_treatment_intent(query) else disease
    tasks.append(fetch_clinical_trials(trial_query, location, limit=8))
    tasks.append(fetch_clinical_trials(disease, "", limit=6))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    publications: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    seen_publications: set[str] = set()
    seen_trials: set[str] = set()

    for result in results:
        if isinstance(result, Exception):
            continue
        for item in result:
            if "nctId" in item:
                item["nearUser"] = item.get("nearUser") or location_matches(location, item)
                trial_id = item.get("nctId")
                if trial_id and trial_id not in seen_trials:
                    seen_trials.add(trial_id)
                    item["relevanceScore"] = score_item(query, item, disease, query)
                    trials.append(item)
            else:
                pub_id = item.get("id")
                if pub_id and pub_id not in seen_publications:
                    seen_publications.add(pub_id)
                    item["relevanceScore"] = score_item(query, item, disease, query)
                    publications.append(item)

    publications = filter_publications(publications, disease, query)
    publications.sort(key=lambda item: item.get("relevanceScore", 0), reverse=True)
    trials.sort(
        key=lambda item: (item.get("nearUser", False), "recruiting" in item.get("status", ""), item.get("relevanceScore", 0)),
        reverse=True,
    )

    payload = {"publications": publications[:8], "trials": trials[:5]}
    set_cached_result(cache_key, payload)
    return payload["publications"], payload["trials"]


async def synthesize(
    query: str,
    publications: list[dict[str, Any]],
    trials: list[dict[str, Any]],
    *,
    question_text: str = "",
    prescription_like: bool = False,
) -> str:
    question_focus = question_text or query
    llm_text = await generate_with_optional_llm(query, publications + trials)
    allow_llm_answer = not is_trial_query(question_focus) and not is_side_effect_query(question_focus)

    if allow_llm_answer and llm_text and llm_answer_looks_relevant(llm_text, question_focus):
        cleaned_llm_text = sanitize_llm_text(llm_text)
        prefix = "I can summarize the evidence, but I cannot recommend a dose or prescribe medication.\n\n" if prescription_like else ""
        suffix = "" if "qualified clinician" in cleaned_llm_text.lower() else "\n\n**Please discuss personal medical decisions with a qualified clinician.**"
        return prefix + cleaned_llm_text + suffix

    if not publications and not trials:
        if has_nutrition_intent(query):
            return "I could not confirm a food-specific answer from the live evidence right now. Please try a more specific nutrition question or check with a clinician or dietitian."
        return "I could not find enough live evidence right now. Please try a narrower disease name or a broader location."

    lines = ["### Quick answer", ""]
    summary_text = " ".join(f"{item.get('title', '')} {item.get('abstract', '')}" for item in publications[:4]).lower()

    if prescription_like:
        lines.append("I cannot recommend a dose, but the evidence suggests this should be discussed with your treating doctor because interactions depend on your cancer therapy.")
    elif is_trial_query(question_focus):
        nearby_trials = sum(1 for trial in trials[:4] if trial.get("nearUser"))
        if trials and nearby_trials:
            lines.append(f"I found relevant clinical trials, including {nearby_trials} that appear closer to your location.")
        elif trials:
            lines.append("I found relevant clinical trials, but the live sources do not clearly confirm a close location match yet.")
        else:
            lines.append("I did not find a clear recruiting trial match from the live sources right now, but I summarized the closest relevant evidence below.")
    elif is_side_effect_query(question_focus):
        caution_points: list[str] = []
        if "pneumonitis" in summary_text or "interstitial lung disease" in summary_text or "lung inflammation" in summary_text:
            caution_points.append("breathing changes or lung inflammation")
        if "immune" in summary_text or "immunotherapy" in summary_text:
            caution_points.append("immune-related reactions")
        if "toxicity" in summary_text or "drug-induced" in summary_text:
            caution_points.append("treatment toxicity monitoring")
        if "interaction" in summary_text or "interactions" in summary_text:
            caution_points.append("medicine interactions")

        if not caution_points:
            caution_points = ["breathing changes", "fatigue", "rash or fever", "possible medicine interactions"]
        lines.append("The strongest evidence suggests discussing " + ", ".join(caution_points[:4]) + " with the care team.")
    elif publications:
        if has_nutrition_intent(query):
            lines.append("I found limited nutrition-related evidence and summarized the most relevant points below. If you want specific diet advice, it should be checked with a clinician or dietitian.")
        else:
            themes: list[str] = []
            if "immunotherapy" in summary_text:
                themes.append("immunotherapy")
            if "targeted" in summary_text:
                themes.append("targeted therapy")
            if "chemotherapy" in summary_text:
                themes.append("chemotherapy")
            if "radiation" in summary_text or "radiotherapy" in summary_text:
                themes.append("radiation therapy")
            if "surgery" in summary_text:
                themes.append("surgery")

            if themes:
                lines.append("Recent evidence points to options such as " + ", ".join(themes[:4]) + ", depending on disease stage and patient profile.")
            else:
                lines.append("I found recent evidence related to your question and summarized the strongest takeaways below.")
    else:
        lines.append("I found some relevant evidence, but it is still limited and should be interpreted carefully.")

    lines.extend(["", "### What the evidence suggests", ""])

    for item in publications[:3]:
        snippet = summarize_snippet(item.get("abstract", ""))
        lines.append(f"- **{item['title']}** — {snippet}")

    if trials:
        lines.extend(["", "### Related trials", ""])
        for trial in trials[:2]:
            lines.append(f"- **{trial['title']}** — status: {trial['status']}")

    lines.extend(["", "**Please discuss personal medical decisions with a qualified clinician.**"])
    return "\n".join(lines)


def build_followups(disease: str, location: str, query: str = "") -> list[str]:
    location_text = f" in {location}" if location else ""
    if has_nutrition_intent(query):
        return [
            f"What nutrition evidence exists for {disease}?",
            f"Are there supplement interactions to discuss for {disease}?",
            f"What diet questions should patients ask a doctor about {disease}?",
        ]
    return [
        f"What are the latest guideline-backed therapies for {disease}?",
        f"Are there recruiting trials for {disease}{location_text}?",
        f"What side effects or interactions should patients discuss with a doctor?",
    ]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "ai"}


@app.post("/v1/research")
async def research(payload: ResearchRequest) -> dict[str, Any]:
    publications, trials = await gather_research(payload.query, payload.query, payload.location)
    return {"publications": publications, "trials": trials}


@app.post("/v1/chat")
async def chat(payload: ChatRequest) -> dict[str, Any]:
    latest_context = " ".join(turn.content for turn in payload.history[-4:])
    focus_topic = resolve_focus_topic(payload.disease, payload.message)

    if is_explicit_topic_switch(payload.disease, payload.message, focus_topic):
        answer = (
            "### Quick answer\n\n"
            f"Your current session is focused on **{payload.disease}**, but your latest question looks like it is about **{focus_topic}**.\n\n"
            "Please switch the Disease field or start a new session if you want evidence for that condition."
        )
        return {
            "answer": answer,
            "citations": [],
            "trials": [],
            "followUps": [
                f"Switch focus to {focus_topic}",
                f"Start a new session for {focus_topic}",
                f"What nutrition evidence exists for {payload.disease}?",
            ],
        }

    scoped_history = latest_context if focus_topic == payload.disease else ""
    query = f"{focus_topic} {payload.message} {payload.location} {scoped_history}".strip()
    publications, trials = await gather_research(query, focus_topic, payload.location, scoped_history)
    if not evidence_is_specific_enough(publications, focus_topic, payload.message):
        publications = []
        trials = []

    answer = await synthesize(
        query,
        publications,
        trials,
        question_text=payload.message,
        prescription_like=is_prescription_request(payload.message),
    )

    return {
        "answer": answer,
        "citations": publications[:6],
        "trials": trials[:4],
        "followUps": build_followups(focus_topic, payload.location, payload.message),
    }
