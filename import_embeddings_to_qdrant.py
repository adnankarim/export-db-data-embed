"""
Import embedding parquet files into Qdrant and link the indexed subset back to MongoDB.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd
from pymongo import MongoClient, UpdateOne
from qdrant_client import QdrantClient, models

from common import (
    DEFAULT_BM25_MODEL,
    DEFAULT_BOOTSTRAP_MODEL,
    DEFAULT_CANDIDATE_ALIAS,
    DEFAULT_CHUNK_ALIAS,
    DEFAULT_DENSE_VECTOR_NAME,
    DEFAULT_SPARSE_VECTOR_NAME,
    physical_collection_name,
)

MONGO_CHUNK_COLLECTION = "candidate_search_chunks"
VECTOR_INDEX_STATE_COLLECTION = "vector_index_state"
ACTIVE_STATE_ID = "active"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import parquet embeddings into Qdrant and Mongo")
    parser.add_argument("--candidate-parquet", required=True)
    parser.add_argument("--chunk-parquet", required=True)
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    parser.add_argument("--mongo-db", default="synthetic_recruitment")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--embedding-model", default=DEFAULT_BOOTSTRAP_MODEL)
    parser.add_argument("--candidate-alias", default=DEFAULT_CANDIDATE_ALIAS)
    parser.add_argument("--chunk-alias", default=DEFAULT_CHUNK_ALIAS)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def _normalize_vector(value) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def _build_sparse_document(text: str) -> models.Document:
    return models.Document(text=text, model=DEFAULT_BM25_MODEL)


def _point_from_row(row: dict) -> models.PointStruct:
    payload = dict(row)
    payload.pop("embedding", None)
    payload.pop("embedding_dim", None)
    payload.pop("target_collection_alias", None)
    return models.PointStruct(
        id=row["qdrant_point_id"],
        vector={
            DEFAULT_DENSE_VECTOR_NAME: _normalize_vector(row["embedding"]),
            DEFAULT_SPARSE_VECTOR_NAME: _build_sparse_document(row["text"]),
        },
        payload=payload,
    )


def _ensure_collection(client: QdrantClient, name: str, vector_size: int, recreate: bool) -> None:
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config={
            DEFAULT_DENSE_VECTOR_NAME: models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            DEFAULT_SPARSE_VECTOR_NAME: models.SparseVectorParams(
                modifier=models.Modifier.IDF
            )
        },
    )


def _swap_alias(client: QdrantClient, alias_name: str, collection_name: str) -> None:
    current_aliases = client.get_aliases().aliases or []
    actions = []
    for alias in current_aliases:
        if alias.alias_name == alias_name:
            actions.append(
                models.DeleteAliasOperation(
                    delete_alias=models.DeleteAlias(alias_name=alias_name)
                )
            )
            break
    actions.append(
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(collection_name=collection_name, alias_name=alias_name)
        )
    )
    client.update_collection_aliases(change_aliases_operations=actions)


def _bulk_upsert(client: QdrantClient, collection_name: str, rows: list[dict], batch_size: int) -> int:
    count = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        points = [_point_from_row(row) for row in batch]
        if points:
            client.upsert(collection_name=collection_name, points=points, wait=True)
            count += len(points)
    return count


def _update_mongo_flags(
    db,
    profil_ids: list[str],
    *,
    recreate: bool,
    embedding_model: str,
    candidate_collection_name: str,
    chunk_collection_name: str,
    batch_size: int,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    candidate_collection = db["candidates"]
    metadata_collection = db["candidate_metadata"]

    if recreate:
        reset_payload = {
            "$set": {
                "vector_indexed": False,
                "vector_indexed_at": None,
                "vector_embedding_model": None,
                "vector_candidate_collection": None,
                "vector_chunk_collection": None,
            }
        }
        candidate_collection.update_many({}, reset_payload)
        metadata_collection.update_many({}, reset_payload)

    candidate_ops = []
    metadata_ops = []
    payload = {
        "$set": {
            "vector_indexed": True,
            "vector_indexed_at": timestamp,
            "vector_embedding_model": embedding_model,
            "vector_candidate_collection": candidate_collection_name,
            "vector_chunk_collection": chunk_collection_name,
        }
    }
    for profil_id in profil_ids:
        candidate_ops.append(UpdateOne({"profil_id": profil_id}, payload))
        metadata_ops.append(UpdateOne({"profil_id": profil_id}, payload))

    for start in range(0, len(candidate_ops), batch_size):
        candidate_collection.bulk_write(candidate_ops[start : start + batch_size], ordered=False)
    for start in range(0, len(metadata_ops), batch_size):
        metadata_collection.bulk_write(metadata_ops[start : start + batch_size], ordered=False)


def _write_chunk_docs(db, rows: list[dict], profil_ids: list[str]) -> None:
    collection = db[MONGO_CHUNK_COLLECTION]
    collection.create_index("chunk_id", unique=True)
    collection.create_index("profil_id")
    collection.create_index([("profil_id", 1), ("layer", 1), ("section_type", 1)])
    collection.create_index("vector_indexed")
    collection.delete_many({"profil_id": {"$in": profil_ids}})

    docs = []
    for row in rows:
        doc = dict(row)
        doc.pop("embedding", None)
        doc.pop("embedding_dim", None)
        doc.pop("target_collection_alias", None)
        doc["vector_indexed"] = True
        docs.append(doc)
    if docs:
        collection.insert_many(docs, ordered=False)


def _update_active_state(
    db,
    *,
    embedding_model: str,
    selected_candidates: int,
    candidate_alias: str,
    candidate_collection: str,
    chunk_alias: str,
    chunk_collection: str,
    stats: dict,
) -> None:
    db[VECTOR_INDEX_STATE_COLLECTION].update_one(
        {"_id": ACTIVE_STATE_ID},
        {
            "$set": {
                "embedding_model": embedding_model,
                "target_count": selected_candidates,
                "diverse_only": True,
                "candidate_alias": candidate_alias,
                "candidate_collection": candidate_collection,
                "chunk_alias": chunk_alias,
                "chunk_collection": chunk_collection,
                "stats": stats,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        },
        upsert=True,
    )


def main() -> None:
    args = parse_args()
    candidate_rows = pd.read_parquet(args.candidate_parquet).to_dict(orient="records")
    chunk_rows = pd.read_parquet(args.chunk_parquet).to_dict(orient="records")
    if not candidate_rows:
        raise RuntimeError("Candidate parquet is empty")

    embedding_model = candidate_rows[0].get("embedding_model") or args.embedding_model
    candidate_collection_name = physical_collection_name(args.candidate_alias, embedding_model)
    chunk_collection_name = physical_collection_name(args.chunk_alias, embedding_model)
    vector_size = len(_normalize_vector(candidate_rows[0]["embedding"]))

    mongo_client = MongoClient(args.mongo_uri)
    db = mongo_client[args.mongo_db]
    qdrant_client = QdrantClient(url=args.qdrant_url)

    _ensure_collection(qdrant_client, candidate_collection_name, vector_size, args.recreate)
    _ensure_collection(qdrant_client, chunk_collection_name, vector_size, args.recreate)

    candidate_count = _bulk_upsert(qdrant_client, candidate_collection_name, candidate_rows, args.batch_size)
    chunk_count = _bulk_upsert(qdrant_client, chunk_collection_name, chunk_rows, args.batch_size)

    all_rows = candidate_rows + chunk_rows
    profil_ids = sorted({row["profil_id"] for row in all_rows})
    _write_chunk_docs(db, all_rows, profil_ids)

    _swap_alias(qdrant_client, args.candidate_alias, candidate_collection_name)
    _swap_alias(qdrant_client, args.chunk_alias, chunk_collection_name)
    _update_mongo_flags(
        db,
        profil_ids,
        recreate=args.recreate,
        embedding_model=embedding_model,
        candidate_collection_name=candidate_collection_name,
        chunk_collection_name=chunk_collection_name,
        batch_size=args.batch_size,
    )

    stats = {
        "embedding_model": embedding_model,
        "selected_candidates": len(profil_ids),
        "candidate_vectors": candidate_count,
        "chunk_vectors": chunk_count,
        "candidate_alias": args.candidate_alias,
        "candidate_collection": candidate_collection_name,
        "chunk_alias": args.chunk_alias,
        "chunk_collection": chunk_collection_name,
        "mongo_indexed_candidates": db["candidate_metadata"].count_documents({"vector_indexed": True}),
        "mongo_chunk_count": db[MONGO_CHUNK_COLLECTION].count_documents({"vector_indexed": True}),
        "qdrant_candidate_count": qdrant_client.count(collection_name=args.candidate_alias, exact=True).count,
        "qdrant_chunk_count": qdrant_client.count(collection_name=args.chunk_alias, exact=True).count,
    }
    _update_active_state(
        db,
        embedding_model=embedding_model,
        selected_candidates=len(profil_ids),
        candidate_alias=args.candidate_alias,
        candidate_collection=candidate_collection_name,
        chunk_alias=args.chunk_alias,
        chunk_collection=chunk_collection_name,
        stats=stats,
    )

    print(json.dumps(stats, indent=2))
    mongo_client.close()


if __name__ == "__main__":
    main()
