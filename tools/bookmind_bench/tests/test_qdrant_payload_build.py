import json
import sys
import types
from pathlib import Path

import qdrant_ingest


class _FakeEmbedModel:
    def encode(self, texts, **kwargs):
        return _FakeVectors([[0.1, 0.2, 0.3, 0.4] for _ in texts])


class _FakeVectors:
    def __init__(self, rows):
        self._rows = rows

    def tolist(self):
        return list(self._rows)


class _FakePointStruct:
    def __init__(self, id, vector, payload):
        self.id = id
        self.vector = vector
        self.payload = payload


class _FakeVectorParams:
    def __init__(self, size, distance):
        self.size = size
        self.distance = distance


class _FakeDistance:
    COSINE = "COSINE"
    DOT = "DOT"
    EUCLID = "EUCLID"
    MANHATTAN = "MANHATTAN"


class _FakeModels:
    PointStruct = _FakePointStruct
    VectorParams = _FakeVectorParams
    Distance = _FakeDistance


class _FakeQdrantClient:
    last_instance = None

    def __init__(self, *args, **kwargs):
        self.upserts = []
        _FakeQdrantClient.last_instance = self

    def collection_exists(self, collection_name):
        return False

    def delete_collection(self, collection_name):
        return None

    def create_collection(self, collection_name, vectors_config):
        return None

    def upsert(self, collection_name, points, wait):
        self.upserts.append((collection_name, points, wait))


def test_qdrant_payload_and_point_id(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "run" / "bundle"
    bundle_dir.mkdir(parents=True)
    records_path = bundle_dir / "records.jsonl"
    records_path.write_text(
        json.dumps(
            {
                "stable_id": "sid-001",
                "page": 2,
                "bbox": [1, 2, 3, 4],
                "content_type": "figure_caption",
                "text": "caption text",
                "meta": {"trace": {"a": 1}, "figure_ref": {"crop_image": "crop.png"}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(qdrant_ingest, "_load_embedding_model", lambda *_: _FakeEmbedModel())

    qdrant_module = types.ModuleType("qdrant_client")
    qdrant_module.QdrantClient = _FakeQdrantClient
    http_module = types.ModuleType("qdrant_client.http")
    models_module = _FakeModels
    http_module.models = models_module

    monkeypatch.setitem(sys.modules, "qdrant_client", qdrant_module)
    monkeypatch.setitem(sys.modules, "qdrant_client.http", http_module)

    stats = qdrant_ingest.ingest_bundle(
        bundle_dir=str(bundle_dir),
        qdrant_url="http://127.0.0.1:6333",
        collection="bookmind_bench",
        embed_model="all-MiniLM-L6-v2",
    )

    assert stats["records_ingested"] == 1

    client = _FakeQdrantClient.last_instance
    assert client is not None
    _, points, _ = client.upserts[0]
    point = points[0]

    assert point.id == "sid-001"
    payload = point.payload
    assert payload["stable_id"] == "sid-001"
    assert payload["page"] == 2
    assert payload["bbox"] == [1, 2, 3, 4]
    assert payload["content_type"] == "figure_caption"
    assert payload["text"] == "caption text"
    assert payload["meta"]["trace"] == {"a": 1}
    assert payload["meta"]["run_dir"] == str(bundle_dir.parent)
    assert payload["meta"]["bundle_relpath"] == "bundle"
