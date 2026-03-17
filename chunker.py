"""
Structure-aware chunking copied for standalone parquet workflows.
"""

from __future__ import annotations

import re
import uuid
from typing import Iterable

CHUNK_NAMESPACE = uuid.UUID("6b0e2a53-36d1-4d60-b9c1-7f7f9ff7a04c")


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        parts = re.split(r"\s*/\s*|,\s*", value)
        return [part.strip() for part in parts if part.strip()]
    return [str(value).strip()]


def _stringify(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def _lowercase_list(values: Iterable[str]) -> list[str]:
    return [value.lower() for value in values if value]


def _safe_float(value) -> float:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return 0.0


def _stable_id(seed: str) -> str:
    return str(uuid.uuid5(CHUNK_NAMESPACE, seed))


def _token_estimate(text: str) -> int:
    return len(text.split())


def _clean_lines(*lines: str) -> list[str]:
    cleaned = []
    for line in lines:
        if line is None:
            continue
        line = str(line).strip()
        if line:
            cleaned.append(line)
    return cleaned


def _extract_synth_blocks(synth_profile: str) -> dict[str, str]:
    if not synth_profile:
        return {}

    blocks: dict[str, str] = {}
    pattern = re.compile(r"\[(?P<key>[^\]]+)\]\s*:\s*(?P<value>.*?)(?=\n\[[^\]]+\]\s*:|\Z)", re.S)
    for match in pattern.finditer(synth_profile):
        key = match.group("key").strip().lower()
        value = match.group("value").strip()
        if value:
            blocks[key] = value
    return blocks


def _split_experience_entries(experience: str) -> list[str]:
    if not experience:
        return []
    parts = re.split(r"\s*/\s*(?=Experience\s+\d+:)", experience.strip())
    return [part.strip() for part in parts if part.strip()]


def _group_experience_entries(entries: list[str], target_tokens: int = 180) -> list[list[str]]:
    if not entries:
        return []

    groups: list[list[str]] = []
    current_group: list[str] = []
    current_tokens = 0

    for entry in entries:
        entry_tokens = _token_estimate(entry)
        if current_group and current_tokens + entry_tokens > target_tokens:
            groups.append(current_group)
            current_group = [entry]
            current_tokens = entry_tokens
        else:
            current_group.append(entry)
            current_tokens += entry_tokens

    if current_group:
        groups.append(current_group)

    return groups


def _clean_location(metadata: dict, candidate: dict) -> str:
    parts = _clean_lines(
        metadata.get("location"),
        metadata.get("country"),
        metadata.get("region"),
    )
    if parts:
        return ", ".join(parts)
    synth_blocks = _extract_synth_blocks(candidate.get("synth_profil", ""))
    return synth_blocks.get("location", "")


def _header_lines(candidate: dict, metadata: dict) -> list[str]:
    return _clean_lines(
        f"Candidate ID: {candidate.get('profil_id')}",
        f"Title: {metadata.get('title') or candidate.get('title')}",
        f"Family: {metadata.get('family') or candidate.get('type')}",
        f"Seniority: {metadata.get('seniority') or candidate.get('profil_experience')}",
        f"Years experience: {_safe_float(metadata.get('years_experience') or candidate.get('nb_year_experiences'))}",
        f"Domains: {_stringify(metadata.get('domains') or candidate.get('domaine'))}",
        f"Location: {_clean_location(metadata, candidate)}",
    )


def _base_payload(candidate: dict, metadata: dict) -> dict:
    skills = _as_list(metadata.get("skills"))
    domains = _as_list(metadata.get("domains") or candidate.get("domaine"))
    languages = _as_list(metadata.get("languages") or candidate.get("languages"))
    certifications = _as_list(metadata.get("certifications") or candidate.get("certification"))

    return {
        "profil_id": candidate.get("profil_id"),
        "title": metadata.get("title") or candidate.get("title"),
        "family": metadata.get("family") or candidate.get("type"),
        "family_lc": str(metadata.get("family") or candidate.get("type") or "").lower(),
        "seniority": metadata.get("seniority") or candidate.get("profil_experience"),
        "seniority_lc": str(
            metadata.get("seniority") or candidate.get("profil_experience") or ""
        ).lower(),
        "years_experience": _safe_float(
            metadata.get("years_experience") or candidate.get("nb_year_experiences")
        ),
        "location": metadata.get("location"),
        "country": metadata.get("country"),
        "country_lc": str(metadata.get("country") or "").lower(),
        "region": metadata.get("region"),
        "region_lc": str(metadata.get("region") or "").lower(),
        "languages": languages,
        "languages_lc": _lowercase_list(languages),
        "domains": domains,
        "domains_lc": _lowercase_list(domains),
        "skills": skills,
        "skills_lc": _lowercase_list(skills),
        "certifications": certifications,
        "certifications_lc": _lowercase_list(certifications),
    }


def _make_chunk(
    candidate: dict,
    metadata: dict,
    *,
    layer: str,
    section_type: str,
    section_ordinal: int,
    text: str,
    local_skills: list[str] | None = None,
) -> dict:
    payload = _base_payload(candidate, metadata)
    chunk_id = _stable_id(
        f"{candidate.get('profil_id')}:{layer}:{section_type}:{section_ordinal}:{text}"
    )
    payload.update(
        {
            "chunk_id": chunk_id,
            "qdrant_point_id": chunk_id,
            "layer": layer,
            "section_type": section_type,
            "section_ordinal": section_ordinal,
            "token_estimate": _token_estimate(text),
            "text": text,
            "local_skills": local_skills or [],
            "local_skills_lc": _lowercase_list(local_skills or []),
        }
    )
    return payload


def build_candidate_chunks(candidate: dict, metadata: dict | None = None) -> list[dict]:
    metadata = metadata or {}
    header = _header_lines(candidate, metadata)
    synth_blocks = _extract_synth_blocks(candidate.get("synth_profil", ""))
    experience_entries = _split_experience_entries(candidate.get("experience", ""))
    experience_groups = _group_experience_entries(experience_entries)
    skills = _as_list(metadata.get("skills"))
    domains = _as_list(metadata.get("domains") or candidate.get("domaine"))
    languages = _as_list(metadata.get("languages") or candidate.get("languages"))
    certifications = _as_list(metadata.get("certifications") or candidate.get("certification"))

    chunks: list[dict] = []

    canonical_lines = header + _clean_lines(
        f"Certifications: {_stringify(certifications)}",
        f"Languages: {_stringify(languages)}",
        "Summary:",
        candidate.get("summary", ""),
        "Profile synthesis:",
        synth_blocks.get("missions") or synth_blocks.get("edge_tags") or candidate.get("synth_profil", ""),
    )
    chunks.append(
        _make_chunk(
            candidate,
            metadata,
            layer="candidate",
            section_type="full_profile",
            section_ordinal=0,
            text="\n".join(canonical_lines),
            local_skills=skills,
        )
    )

    section_specs = [
        ("identity_header", _clean_lines(*header)),
        ("summary", header + _clean_lines("Section: Summary", candidate.get("summary", ""))),
        (
            "skills_domains",
            header
            + _clean_lines(
                "Section: Skills and domains",
                f"Skills: {_stringify(skills) or candidate.get('skills', '')}",
                f"Domains: {_stringify(domains)}",
                f"Certifications: {_stringify(certifications)}",
                f"Languages: {_stringify(languages)}",
            ),
        ),
        (
            "education",
            header
            + _clean_lines(
                "Section: Education",
                f"Main diploma: {candidate.get('main_diploma', '')}",
                f"Diplomas: {candidate.get('diplomas', '')}",
                f"Schools: {candidate.get('schools', '')}",
            ),
        ),
        (
            "languages_location",
            header
            + _clean_lines(
                "Section: Languages and location",
                f"Languages: {_stringify(languages) or candidate.get('languages', '')}",
                f"Location: {_clean_location(metadata, candidate)}",
                f"Region: {metadata.get('region')}",
            ),
        ),
        (
            "synth_profile",
            header + _clean_lines("Section: Profile synthesis", candidate.get("synth_profil", "")),
        ),
    ]

    for section_ordinal, (section_type, lines) in enumerate(section_specs):
        if len(lines) <= len(header):
            continue
        chunks.append(
            _make_chunk(
                candidate,
                metadata,
                layer="section",
                section_type=section_type,
                section_ordinal=section_ordinal,
                text="\n".join(lines),
                local_skills=skills,
            )
        )

    for group_idx, group_entries in enumerate(experience_groups):
        chunks.append(
            _make_chunk(
                candidate,
                metadata,
                layer="section",
                section_type="experience_group",
                section_ordinal=group_idx,
                text="\n".join(header + _clean_lines("Section: Experience", *group_entries)),
                local_skills=skills,
            )
        )

    for entry_idx, entry in enumerate(experience_entries):
        chunks.append(
            _make_chunk(
                candidate,
                metadata,
                layer="evidence",
                section_type="experience_evidence",
                section_ordinal=entry_idx,
                text="\n".join(header + _clean_lines("Section: Experience evidence", entry)),
                local_skills=skills,
            )
        )

    return chunks
