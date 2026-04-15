from __future__ import annotations

import httpx


async def fetch_clinical_trials(query: str, location: str = "", limit: int = 5) -> list[dict]:
    params = {"query.term": query, "pageSize": limit}
    if location:
        params["query.locn"] = location

    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            response = await client.get("https://clinicaltrials.gov/api/query/studies", params=params)
            response.raise_for_status()
            payload = response.json()
    except Exception:
        return []

    studies = payload.get("studies") or payload.get("StudyFieldsResponse", {}).get("StudyFields") or []
    results = []
    for study in studies[:limit]:
        protocol = study.get("protocolSection", {}) if isinstance(study, dict) else {}
        identification = protocol.get("identificationModule", {})
        status_module = protocol.get("statusModule", {})
        conditions_module = protocol.get("conditionsModule", {})
        arms_module = protocol.get("armsInterventionsModule", {})
        contacts_module = protocol.get("contactsLocationsModule", {})

        nct_id = identification.get("nctId") or ""
        title = identification.get("briefTitle") or study.get("BriefTitle", [""])[0]
        status = status_module.get("overallStatus") or study.get("OverallStatus", ["unknown"])[0]
        phase_list = protocol.get("designModule", {}).get("phases") or study.get("Phase", [])
        phase = phase_list[0] if phase_list else "na"
        conditions = conditions_module.get("conditions") or study.get("Condition", [])
        interventions = [item.get("name", "") for item in arms_module.get("interventions", [])] or study.get("InterventionName", [])

        locations = []
        for loc in contacts_module.get("locations", [])[:3]:
            facility = loc.get("facility", "")
            city = loc.get("city", "")
            country = loc.get("country", "")
            locations.append({"facility": facility, "city": city, "country": country})

        if not nct_id and isinstance(study, dict):
            nct_id = (study.get("NCTId") or [""])[0]

        if nct_id:
            results.append(
                {
                    "nctId": nct_id,
                    "title": title,
                    "status": str(status).lower(),
                    "phase": str(phase).lower(),
                    "conditions": conditions,
                    "interventions": interventions,
                    "locations": locations,
                    "url": f"https://clinicaltrials.gov/study/{nct_id}",
                    "nearUser": bool(location and any(location.lower() in (f'{loc.get('city', '')} {loc.get('country', '')}').lower() for loc in locations)),
                }
            )

    return results
