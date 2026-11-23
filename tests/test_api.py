"""The FastAPI service with tiny fake models: no weights, data or GPU needed (so it runs in CI)."""
import io

import faiss
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import api

N = 300                                                      # > the API's search pool of 200
IDS = [f"0100000{i:03d}" for i in range(N)]
GROUPS = ["Garment Upper body", "Garment Lower body", "Underwear"] * (N // 3)


class FakeBoxes:
    xyxy = np.array([[10.0, 10.0, 200.0, 300.0], [50.0, 320.0, 300.0, 580.0]])
    conf = np.array([0.91, 0.84])
    cls = np.array([0.0, 1.0])


class FakeYOLO:
    """Mimics ultralytics: .predict() -> [result with .boxes and .names]."""
    def __init__(self, *_):
        pass

    def predict(self, *_, **__):
        r = type("Result", (), {})()
        r.boxes, r.names = FakeBoxes(), {0: "long sleeve top", 1: "skirt"}
        return [r]


def unit(rng, *shape):
    v = rng.normal(size=shape).astype("float32")
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def fake_index(alpha):
    """Built exactly like predict.build_index, from random 4-number 'fingerprints'."""
    rng = np.random.default_rng(0)
    X = np.hstack([unit(rng, N, 4) * np.sqrt(alpha), unit(rng, N, 4) * np.sqrt(1 - alpha)]).astype("float32")
    index = faiss.IndexFlatIP(8)
    index.add(X)
    return index, {"ids": IDS, "group": GROUPS, "colour": ["Blue"] * N}


def fake_run_models(tagger, designer, x, device):
    rng = np.random.default_rng(1)
    return None, None, unit(rng, 4), unit(rng, 4)               # colour/type fingerprint, design fingerprint


def fake_tags(out_g, out_c, top):
    return {"group": [("Garment Upper body", 0.9)], "colour": [("Blue", 0.8)]}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "YOLO", FakeYOLO)
    monkeypatch.setattr(api, "load_models", lambda device: (None, None))
    monkeypatch.setattr(api, "build_index", fake_index)
    monkeypatch.setattr(api, "load_recommender",
                        lambda device: {"model": None, "pop": IDS[:12], "art2idx": {}, "idx2art": {}, "kind": "lstm"})
    monkeypatch.setattr(api, "run_models", fake_run_models)
    monkeypatch.setattr(api, "tags", fake_tags)
    (tmp_path / "010").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "010" / "0100000001.jpg")
    monkeypatch.setattr(api, "IMG_DIR", tmp_path)
    with TestClient(api.app) as c:                              # "with" runs the startup (lifespan)
        yield c


def jpeg(size=(400, 600)):
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 50, 50)).save(buf, format="JPEG")
    return {"image": ("photo.jpg", buf.getvalue(), "image/jpeg")}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "device": api.M["device"], "catalogue_size": N}


def test_predict_returns_every_garment(client):
    r = client.post("/predict?k=5", files=jpeg())
    assert r.status_code == 200
    body = r.json()
    assert body["n_garments"] == 2 and len(body["garments"]) == 2
    g = body["garments"][0]
    assert g["label"] == "long sleeve top" and g["box"] == [10.0, 10.0, 200.0, 300.0]
    assert g["tags"]["group"] == "Garment Upper body"
    assert len(g["similar"]) == 5
    assert body["garments"][1]["tags"]["group_source"] == "from detector"   # skirt -> Lower body hint


def test_sensitive_groups_hidden_by_default(client):
    body = client.post("/predict?k=20", files=jpeg()).json()
    groups = {p["group"] for g in body["garments"] for p in g["similar"] + g["buy_next"]}
    assert "Underwear" not in groups


def test_sensitive_groups_shown_when_asked(client):
    body = client.post("/predict?k=50&hide_sensitive=false&detect=false", files=jpeg()).json()
    assert "Underwear" in {p["group"] for p in body["garments"][0]["similar"]}


def test_no_detection_uses_the_full_photo(client):
    body = client.post("/predict?detect=false", files=jpeg()).json()
    assert body["n_garments"] == 1 and body["garments"][0]["box"] is None


def test_not_an_image_is_400(client):
    r = client.post("/predict", files={"image": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400


@pytest.mark.parametrize("query", ["k=0", "k=999", "match=pink", "detect=maybe"])
def test_bad_parameters_are_422(client, query):
    assert client.post(f"/predict?{query}", files=jpeg()).status_code == 422


def test_missing_file_is_422(client):
    assert client.post("/predict").status_code == 422


def test_product_image(client):
    assert client.get("/products/0100000001/image").status_code == 200
    assert client.get("/products/0100000002/image").status_code == 404     # no photo on disk
    assert client.get("/products/abc/image").status_code == 400             # not a 10-digit id
