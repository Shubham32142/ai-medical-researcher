from __future__ import annotations

import asyncio
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
GENERIC_PENALTY_TERMS = {
    "global burden",
    "incidence",
    "mortality",
    "years of life lost",
    "disability-adjusted",
    "epidemiology",
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


def build_query_variants(disease: str, message: str, location: str, history: str) -> list[str]:
    disease_phrase = f'"{disease}"' if " " in disease else disease
    message_terms = " ".join(extract_keywords(message)[:4])
    history_terms = " ".join(extract_keywords(history)[:3])
    treatment_suffix = "treatment therapy" if has_treatment_intent(message) else "clinical evidence"

    candidates = [
        f"{disease_phrase} {message_terms} {treatment_suffix}".strip(),
        f"{disease_phrase} treatment therapy review".strip(),
        f"{disease_phrase} immunotherapy chemotherapy clinical trial".strip(),
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
    intent_bonus = intent_hits * 1.2 if has_treatment_intent(message) else 0

    recency = max(0, int(item.get("year", 2024)) - 2017) / 10
    source_bonus = 0.55 if item.get("source") == "pubmed" else 0.15
    trial_bonus = 0.7 if item.get("nearUser") else 0
    abstract_bonus = 0.15 if item.get("abstract") else 0

    lowered_text = text.lower()
    generic_penalty = 0
    if any(term in lowered_text for term in GENERIC_PENALTY_TERMS) and not any(term in lowered_text for term in TREATMENT_TERMS):
        generic_penalty = 3.0

    return round((overlap * 1.2) + disease_bonus + intent_bonus + recency + source_bonus + trial_bonus + abstract_bonus - generic_penalty, 3)


def filter_publications(publications: list[dict[str, Any]], disease: str, message: str) -> list[dict[str, Any]]:
    disease_terms = extract_keywords(disease)
    require_treatment = has_treatment_intent(message)
    filtered: list[dict[str, Any]] = []

    for item in publications:
        if int(item.get("year", 2024)) < 2018:
            continue

        lowered_text = f"{item.get('title', '')} {item.get('abstract', '')}".lower()
        disease_hits = sum(1 for term in disease_terms if term in lowered_text)
        if disease_hits == 0:
            continue

        if require_treatment:
            if any(term in lowered_text for term in TREATMENT_TERMS) or "review" in lowered_text:
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


async def synthesize(query: str, publications: list[dict[str, Any]], trials: list[dict[str, Any]], *, prescription_like: bool = False) -> str:
    llm_text = await generate_with_optional_llm(query, publications + trials)
    evidence_ids = [item.get("id") for item in publications[:4]] + [item.get("nctId") for item in trials[:3]]

    if llm_text and any(identifier and identifier in llm_text for identifier in evidence_ids):
        prefix = "I can summarize the evidence, but I cannot recommend a dose or prescribe medication.\n\n" if prescription_like else ""
        return prefix + llm_text.strip() + "\n\n**Please discuss treatment decisions with a qualified clinician.**"

    if not publications and not trials:
        return "I could not find enough live evidence right now. Please try a narrower disease name or a broader location."

    lines = ["### Quick answer", ""]

    if prescription_like:
        lines.append("I cannot recommend a dose, but the evidence suggests this should be discussed with your treating doctor because interactions depend on your cancer therapy.")
    elif publications:
        summary_text = " ".join(f"{item.get('title', '')} {item.get('abstract', '')}" for item in publications[:4]).lower()
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
        snippet = item.get("abstract", "")[:150].strip()
        lines.append(f"- **{item['title']}** — {snippet}")

    if trials:
        lines.extend(["", "### Related trials", ""])
        for trial in trials[:2]:
            lines.append(f"- **{trial['title']}** — status: {trial['status']}")

    lines.extend(["", "**Please discuss treatment decisions with a qualified clinician.**"])
    return "\n".join(lines)


def build_followups(disease: str, location: str) -> list[str]:
    location_text = f" in {location}" if location else ""
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
    query = f"{payload.disease} {payload.message} {payload.location} {latest_context}".strip()
    publications, trials = await gather_research(query, payload.disease, payload.location, latest_context)
    answer = await synthesize(
        query,
        publications,
        trials,
        prescription_like=is_prescription_request(payload.message),
    )

    return {
        "answer": answer,
        "citations": publications[:6],
        "trials": trials[:4],
        "followUps": build_followups(payload.disease, payload.location),
    }
