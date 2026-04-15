from __future__ import annotations

import httpx


def _reconstruct_abstract(index: dict | None) -> str:
    if not index:
        return ""
    ordered: list[tuple[int, str]] = []
    for word, positions in index.items():
        for position in positions:
            ordered.append((position, word))
    ordered.sort(key=lambda item: item[0])
    return " ".join(word for _, word in ordered)


async def fetch_openalex(query: str, limit: int = 8) -> list[dict]:
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            response = await client.get(
                "https://api.openalex.org/works",
                params={"search": query, "per-page": limit, "mailto": "dev@curalink.app"},
            )
            response.raise_for_status()
    except Exception:
        return []

    results = []
    for item in response.json().get("results", [])[:limit]:
        authors = [a.get("author", {}).get("display_name", "") for a in item.get("authorships", [])[:5]]
        results.append(
            {
                "id": item.get("id", "").split("/")[-1] or "openalex",
                "source": "openalex",
                "title": item.get("display_name", "Untitled"),
                "authors": [a for a in authors if a],
                "journal": item.get("primary_location", {}).get("source", {}).get("display_name", ""),
                "year": item.get("publication_year") or 2024,
                "abstract": _reconstruct_abstract(item.get("abstract_inverted_index")),
                "doi": item.get("doi") or "",
                "url": item.get("primary_location", {}).get("landing_page_url") or item.get("id", "https://openalex.org"),
            }
        )
    return results
