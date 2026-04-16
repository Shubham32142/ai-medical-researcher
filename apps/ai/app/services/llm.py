import os
from typing import Any

import httpx


async def generate_with_optional_llm(query: str, evidence: list[dict[str, Any]]) -> str | None:
    prompt = build_prompt(query, evidence)

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        return await call_groq(prompt, groq_key)

    ollama_base_url = os.getenv("OLLAMA_BASE_URL")
    if ollama_base_url:
        return await call_ollama(prompt, ollama_base_url, os.getenv("OLLAMA_MODEL", "llama3.1:8b"))

    return None


async def call_groq(prompt: str, api_key: str) -> str | None:
    async with httpx.AsyncClient(timeout=40) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a medical research assistant. Only answer from the provided evidence. Keep the reply clear, short, and human-friendly. Do not show raw citation IDs like W123 or PMID:123 inside the prose. The UI will display the evidence separately. Avoid direct prescriptions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            },
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"]


async def call_ollama(prompt: str, base_url: str, model: str) -> str | None:
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{base_url.rstrip('/')}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("response")


def build_prompt(query: str, evidence: list[dict[str, Any]]) -> str:
    snippets = []
    for item in evidence[:8]:
        label = item.get("id") or item.get("nctId") or "source"
        text = item.get("abstract") or item.get("title") or ""
        snippets.append(f"[{label}] {item.get('title', 'Untitled')}: {text[:700]}")

    return (
        f"Question: {query}\n\n"
        "You are a medical research assistant. Use only the evidence below."
        " Answer the user's specific intent first."
        " If they ask about clinical trials, address trials first."
        " If they ask about side effects or interactions, focus on adverse effects and cautions first."
        " Write a direct, user-friendly answer followed by a brief evidence summary."
        " Do not print raw citation IDs inside the answer body because the UI shows the evidence separately."
        " If evidence is weak or missing, say so clearly. Refuse dosage or prescription advice.\n\nEvidence:\n"
        + "\n".join(snippets)
    )
