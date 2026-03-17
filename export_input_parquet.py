"""
Export selected synthetic candidates and chunk payloads to parquet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from pymongo import MongoClient

from chunker import build_candidate_chunks
from common import (
    DEFAULT_BOOTSTRAP_MODEL,
    DEFAULT_CANDIDATE_ALIAS,
    DEFAULT_CHUNK_ALIAS,
    DEFAULT_DENSE_VECTOR_NAME,
    DEFAULT_SPARSE_VECTOR_NAME,
    choose_diverse_profile_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export standalone parquet input for synthetic embeddings")
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    parser.add_argument("--mongo-db", default="synthetic_recruitment")
    parser.add_argument("--output-dir", default="standalone_output")
    parser.add_argument("--target-count", type=int, default=1000)
    parser.add_argument("--diverse-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--embedding-model", default=DEFAULT_BOOTSTRAP_MODEL)
    parser.add_argument("--candidate-alias", default=DEFAULT_CANDIDATE_ALIAS)
    parser.add_argument("--chunk-alias", default=DEFAULT_CHUNK_ALIAS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = MongoClient(args.mongo_uri)
    db = client[args.mongo_db]

    metadata_docs = list(
        db["candidate_metadata"].find(
            {},
            {
                "_id": 0,
                "profil_id": 1,
                "family": 1,
                "country": 1,
                "seniority": 1,
            },
        )
    )
    if args.diverse_only:
        selected_ids = choose_diverse_profile_ids(metadata_docs, args.target_count)
    else:
        selected_ids = sorted(doc["profil_id"] for doc in metadata_docs)[: args.target_count]

    candidate_docs = list(db["candidates"].find({"profil_id": {"$in": selected_ids}}, {"_id": 0}))
    candidate_map = {doc["profil_id"]: doc for doc in candidate_docs}
    metadata_map = {
        doc["profil_id"]: doc
        for doc in db["candidate_metadata"].find({"profil_id": {"$in": selected_ids}}, {"_id": 0})
    }

    ordered_candidates = [candidate_map[profil_id] for profil_id in selected_ids if profil_id in candidate_map]

    selected_rows = []
    chunk_rows = []
    for candidate in ordered_candidates:
        profil_id = candidate["profil_id"]
        metadata = metadata_map.get(profil_id, {})
        selected_rows.append(
            {
                "profil_id": profil_id,
                "candidate_json": json.dumps(candidate, ensure_ascii=True, sort_keys=True),
                "metadata_json": json.dumps(metadata, ensure_ascii=True, sort_keys=True),
                "title": metadata.get("title") or candidate.get("title"),
                "family": metadata.get("family") or candidate.get("type"),
                "seniority": metadata.get("seniority") or candidate.get("profil_experience"),
                "country": metadata.get("country"),
                "region": metadata.get("region"),
                "years_experience": metadata.get("years_experience") or candidate.get("nb_year_experiences"),
            }
        )

        for chunk_sort_order, chunk in enumerate(build_candidate_chunks(candidate, metadata)):
            row = dict(chunk)
            row["chunk_sort_order"] = chunk_sort_order
            row["embedding_model"] = args.embedding_model
            row["dense_vector_name"] = DEFAULT_DENSE_VECTOR_NAME
            row["sparse_vector_name"] = DEFAULT_SPARSE_VECTOR_NAME
            row["target_collection_alias"] = (
                args.candidate_alias if row["layer"] == "candidate" else args.chunk_alias
            )
            row["vector_indexed"] = False
            chunk_rows.append(row)

    selected_path = output_dir / "selected_candidates.parquet"
    chunk_input_path = output_dir / "chunk_input.parquet"
    manifest_path = output_dir / "manifest.json"

    pd.DataFrame(selected_rows).to_parquet(selected_path, index=False)
    pd.DataFrame(chunk_rows).to_parquet(chunk_input_path, index=False)

    manifest = {
        "mongo_db": args.mongo_db,
        "target_count": args.target_count,
        "selected_candidates": len(selected_rows),
        "chunk_rows": len(chunk_rows),
        "embedding_model": args.embedding_model,
        "candidate_alias": args.candidate_alias,
        "chunk_alias": args.chunk_alias,
        "selected_candidates_parquet": str(selected_path.name),
        "chunk_input_parquet": str(chunk_input_path.name),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="ascii")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
