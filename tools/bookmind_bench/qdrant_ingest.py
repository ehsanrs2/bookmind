"""Qdrant ingestion + retrieval helpers for bench bundles."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBED_ALIAS = "all-MiniLM-L6-v2"
DEFAULT_EMBED_DIM = 384


def _is_offline_enabled() -> bool:
    values = {
        str(os.environ.get("HF_HUB_OFFLINE", "")).strip().lower(),
        str(os.environ.get("TRANSFORMERS_OFFLINE", "")).strip().lower(),
    }
    return bool(values.intersection({"1", "true", "yes", "y"}))


def _bench_root() -> Path:
    return Path(__file__).resolve().parent


def _default_embedding_dir(model_name: str) -> Path:
    return _bench_root() / "offline_bundle" / "models" / "embeddings" / model_name


def _canonical_model_id(model_name: str) -> str:
    value = str(model_name or "").strip()
    if not value or value == DEFAULT_EMBED_ALIAS:
        return DEFAULT_EMBED_MODEL
    return value


def _resolve_model_ref(embed_model: str) -> str:
    model_ref = str(embed_model or "").strip()
    if not model_ref:
        model_ref = DEFAULT_EMBED_MODEL

    as_path = Path(model_ref)
    if as_path.exists():
        return str(as_path)

    canonical = _canonical_model_id(model_ref)
    model_name = canonical.rsplit("/", 1)[-1]
    local_dir = _default_embedding_dir(model_name)
    if local_dir.exists():
        return str(local_dir)

    if _is_offline_enabled():
        raise RuntimeError(
            "Embedding model unavailable in offline mode. "
            f"Expected local model at: {local_dir}. "
            "Place a SentenceTransformer snapshot there or pass --embed_model to a local path."
        )

    return canonical


def _load_embedding_model(embed_model: str):
    resolved = _resolve_model_ref(embed_model)
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - exercised by integration environment
        raise RuntimeError(
            "sentence-transformers is required for embedding. Install the qdrant profile requirements."
        ) from exc

    try:
        return SentenceTransformer(resolved)
    except Exception as exc:
        if _is_offline_enabled():
            raise RuntimeError(
                f"Failed to load embedding model in offline mode from '{resolved}'. "
                "Ensure model files are present under offline_bundle/models/embeddings/"
                " or pass --embed_model with a local model path."
            ) from exc
        raise


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _chunks(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    if size <= 0:
        size = 1
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def _build_payload(record: Dict[str, Any], run_dir: str, bundle_relpath: str) -> Dict[str, Any]:
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    payload_meta = dict(meta)
    payload_meta["run_dir"] = run_dir
    payload_meta["bundle_relpath"] = bundle_relpath
    return {
        "stable_id": record.get("stable_id"),
        "page": record.get("page"),
        "bbox": record.get("bbox"),
        "content_type": record.get("content_type"),
        "text": record.get("text"),
        "meta": payload_meta,
    }


def ensure_collection(
    client,
    collection: str,
    dim: int,
    distance: str = "Cosine",
    recreate: bool = False,
) -> None:
    """Ensure target collection exists with the expected vector params."""
    try:
        from qdrant_client.http import models
    except Exception as exc:  # pragma: no cover - exercised by integration environment
        raise RuntimeError(
            "qdrant-client is required. Install the qdrant profile requirements."
        ) from exc

    exists = bool(client.collection_exists(collection_name=collection))
    if recreate and exists:
        client.delete_collection(collection_name=collection)
        exists = False

    if exists:
        return

    distance_name = str(distance or "Cosine").strip().upper()
    if distance_name not in {"COSINE", "DOT", "EUCLID", "MANHATTAN"}:
        distance_name = "COSINE"

    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(
            size=int(dim),
            distance=getattr(models.Distance, distance_name),
        ),
    )


def embed_texts(model, texts: Sequence[str], batch_size: int = 64) -> List[List[float]]:
    """Encode texts into normalized vectors."""
    cleaned = [str(text or "") for text in texts]
    if not cleaned:
        return []

    vectors = model.encode(
        cleaned,
        batch_size=max(1, int(batch_size)),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def ingest_bundle(
    bundle_dir: str,
    qdrant_url: str,
    collection: str,
    embed_model: str,
    recreate: bool = False,
    batch_size: int = 64,
    timeout_s: int = 60,
) -> Dict[str, Any]:
    """Ingest normalized bundle records into Qdrant."""
    bundle_path = Path(bundle_dir)
    records_path = bundle_path / "records.jsonl"
    if not records_path.exists():
        raise RuntimeError(f"Bundle records file not found: {records_path}")

    rows = _read_jsonl(records_path)
    records: List[Dict[str, Any]] = []
    skipped_missing_id = 0

    for row in rows:
        stable_id = row.get("stable_id")
        if not isinstance(stable_id, str) or not stable_id.strip():
            skipped_missing_id += 1
            continue
        records.append(row)

    if not records:
        return {
            "records_total": len(rows),
            "records_ingested": 0,
            "records_skipped_missing_id": skipped_missing_id,
            "collection": collection,
            "qdrant_url": qdrant_url,
            "bundle_dir": str(bundle_path),
        }

    model = _load_embedding_model(embed_model)
    vectors = embed_texts(model, [str(r.get("text") or "") for r in records], batch_size=batch_size)
    if not vectors:
        raise RuntimeError("No embeddings were produced from bundle records.")

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models
    except Exception as exc:  # pragma: no cover - exercised by integration environment
        raise RuntimeError(
            "qdrant-client is required. Install the qdrant profile requirements."
        ) from exc

    client = QdrantClient(url=qdrant_url, timeout=timeout_s)
    ensure_collection(
        client=client,
        collection=collection,
        dim=len(vectors[0]),
        distance="Cosine",
        recreate=recreate,
    )

    run_dir = str(bundle_path.parent)
    bundle_relpath = str(bundle_path.name)
    point_count = 0

    indexed: List[tuple[Dict[str, Any], List[float]]] = list(zip(records, vectors))
    for batch in _chunks(indexed, max(1, int(batch_size))):
        points = [
            models.PointStruct(
                id=record["stable_id"],
                vector=vector,
                payload=_build_payload(record, run_dir=run_dir, bundle_relpath=bundle_relpath),
            )
            for record, vector in batch
        ]
        client.upsert(collection_name=collection, points=points, wait=True)
        point_count += len(points)

    return {
        "records_total": len(rows),
        "records_ingested": point_count,
        "records_skipped_missing_id": skipped_missing_id,
        "collection": collection,
        "qdrant_url": qdrant_url,
        "bundle_dir": str(bundle_path),
        "embedding_model": _resolve_model_ref(embed_model),
        "embedding_dim": len(vectors[0]),
    }


def search_query(
    query: str,
    qdrant_url: str,
    collection: str,
    embed_model: str,
    top_k: int = 10,
    timeout_s: int = 30,
    content_types: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Search a Qdrant collection using an embedded text query."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models
    except Exception as exc:  # pragma: no cover - exercised by integration environment
        raise RuntimeError(
            "qdrant-client is required. Install the qdrant profile requirements."
        ) from exc

    model = _load_embedding_model(embed_model)
    vector = embed_texts(model, [query], batch_size=1)[0]
    client = QdrantClient(url=qdrant_url, timeout=timeout_s)

    query_filter = None
    cleaned_types = [str(t).strip() for t in (content_types or []) if str(t).strip()]
    if cleaned_types:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="content_type",
                    match=models.MatchAny(any=cleaned_types),
                )
            ]
        )

    hits = client.search(
        collection_name=collection,
        query_vector=vector,
        limit=max(1, int(top_k)),
        with_payload=True,
        query_filter=query_filter,
    )

    out: List[Dict[str, Any]] = []
    for hit in hits:
        payload = hit.payload if isinstance(hit.payload, dict) else {}
        out.append(
            {
                "id": hit.id,
                "score": float(hit.score),
                "stable_id": payload.get("stable_id"),
                "page": payload.get("page"),
                "bbox": payload.get("bbox"),
                "content_type": payload.get("content_type"),
                "text": payload.get("text"),
                "meta": payload.get("meta") if isinstance(payload.get("meta"), dict) else {},
            }
        )

    return out
