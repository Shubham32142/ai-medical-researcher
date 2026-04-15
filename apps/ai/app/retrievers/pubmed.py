from __future__ import annotations

import xml.etree.ElementTree as ET

import httpx


async def fetch_pubmed(query: str, limit: int = 8) -> list[dict]:
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    try:
        async with httpx.AsyncClient(timeout=25, follow_redirects=True) as client:
            search = await client.get(
                search_url,
                params={"db": "pubmed", "term": query, "retmax": limit, "retmode": "json", "sort": "relevance"},
            )
            search.raise_for_status()
            ids = search.json().get("esearchresult", {}).get("idlist", [])[:limit]
            if not ids:
                return []

            fetched = await client.get(
                fetch_url,
                params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
            )
            fetched.raise_for_status()
    except Exception:
        return []

    root = ET.fromstring(fetched.text)
    articles: list[dict] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID", default="")
        title_node = article.find(".//ArticleTitle")
        title = "".join(title_node.itertext()).strip() if title_node is not None else "Untitled"
        abstract_parts = ["".join(node.itertext()).strip() for node in article.findall(".//AbstractText")]
        abstract = " ".join(part for part in abstract_parts if part)
        authors = []
        for author in article.findall(".//Author")[:5]:
            last = author.findtext("LastName", default="")
            fore = author.findtext("ForeName", default="")
            name = f"{fore} {last}".strip()
            if name:
                authors.append(name)
        journal = article.findtext(".//Journal/Title", default="")
        year = article.findtext(".//PubDate/Year") or article.findtext(".//ArticleDate/Year") or "2024"
        doi = ""
        for identifier in article.findall(".//ArticleId"):
            if identifier.attrib.get("IdType") == "doi":
                doi = identifier.text or ""
                break

        articles.append(
            {
                "id": f"PMID:{pmid}",
                "source": "pubmed",
                "title": title,
                "authors": authors,
                "journal": journal,
                "year": int(year) if str(year).isdigit() else 2024,
                "abstract": abstract,
                "doi": doi,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
        )

    return articles
