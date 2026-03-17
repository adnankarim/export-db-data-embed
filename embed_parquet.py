"""
Embed chunk parquet on a GPU machine and save embedding parquet files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import DEFAULT_BOOTSTRAP_MODEL, StandaloneEmbedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate embedding parquet files from chunk input parquet")
    parser.add_argument("--input-parquet", required=True)
    parser.add_argument("--output-dir", default="embedded_output")
    parser.add_argument("--model-name", default=DEFAULT_BOOTSTRAP_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--candidate-batch-size", type=int, default=32)
    return parser.parse_args()


def _records(df: pd.DataFrame) -> list[dict]:
    return df.to_dict(orient="records")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input_parquet)
    if df.empty:
        raise RuntimeError("Input parquet is empty")

    embedder = StandaloneEmbedder(args.model_name, device=args.device)
    embedder.prefetch()

    candidate_df = df[df["layer"] == "candidate"].copy()
    chunk_df = df[df["layer"] != "candidate"].copy()

    candidate_records = _records(candidate_df.sort_values(["profil_id", "chunk_sort_order"]))
    chunk_records = _records(chunk_df.sort_values(["profil_id", "chunk_sort_order"]))

    candidate_output = []
    for start in range(0, len(candidate_records), args.candidate_batch_size):
        batch = candidate_records[start : start + args.candidate_batch_size]
        vectors = embedder.encode_documents([row["text"] for row in batch])
        for row, vector in zip(batch, vectors):
            row_out = dict(row)
            row_out["embedding"] = vector
            row_out["embedding_dim"] = len(vector)
            candidate_output.append(row_out)

    chunk_output = []
    grouped_chunks: dict[str, list[dict]] = {}
    for row in chunk_records:
        grouped_chunks.setdefault(row["profil_id"], []).append(row)

    grouped_items = list(grouped_chunks.items())
    for start in range(0, len(grouped_items), args.candidate_batch_size):
        batch_groups = grouped_items[start : start + args.candidate_batch_size]
        contextual_inputs = [[row["text"] for row in rows] for _, rows in batch_groups]
        contextual_vectors = embedder.encode_contextual_documents(contextual_inputs)
        for (_, rows), vectors in zip(batch_groups, contextual_vectors):
            for row, vector in zip(rows, vectors):
                row_out = dict(row)
                row_out["embedding"] = vector
                row_out["embedding_dim"] = len(vector)
                chunk_output.append(row_out)

    candidate_output_path = output_dir / "candidate_embeddings.parquet"
    chunk_output_path = output_dir / "chunk_embeddings.parquet"
    manifest_path = output_dir / "embedding_manifest.json"

    pd.DataFrame(candidate_output).to_parquet(candidate_output_path, index=False)
    pd.DataFrame(chunk_output).to_parquet(chunk_output_path, index=False)

    manifest = {
        "model_name": args.model_name,
        "device": args.device,
        "candidate_rows": len(candidate_output),
        "chunk_rows": len(chunk_output),
        "candidate_output_parquet": str(candidate_output_path.name),
        "chunk_output_parquet": str(chunk_output_path.name),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="ascii")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
