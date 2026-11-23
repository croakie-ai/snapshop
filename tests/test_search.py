"""Hybrid search on a tiny fake catalogue (no real embeddings needed)."""
import faiss
import numpy as np
import pytest

from predict import TAG_TRUST, final_group, search


def unit(rng, n, d):
    v = rng.normal(size=(n, d)).astype("float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture
def catalogue():
    rng = np.random.default_rng(0)
    design, colour = unit(rng, 20, 8), unit(rng, 20, 6)               # 20 fake products
    meta = {"group": ["Garment Upper body"] * 10 + ["Garment Lower body"] * 10}
    return design, colour, meta


def index_for(design, colour, alpha):
    """Same construction as predict.build_index: sqrt-weighted parts side by side."""
    X = np.hstack([design * np.sqrt(alpha), colour * np.sqrt(1 - alpha)]).astype("float32")
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    return index


@pytest.mark.parametrize("alpha", [1.0, 0.5, 0.25])
def test_a_catalogue_product_finds_itself_first(catalogue, alpha):
    design, colour, _ = catalogue
    scores, idx = search(index_for(design, colour, alpha), design[3], colour[3], alpha, 5)
    assert idx[0] == 3
    assert scores[0] == pytest.approx(1.0, abs=1e-5)                  # cosine with itself = 1


def test_score_is_the_weighted_mix_of_two_cosines(catalogue):
    design, colour, _ = catalogue
    rng = np.random.default_rng(1)
    qd, qc = unit(rng, 1, 8)[0], unit(rng, 1, 6)[0]
    alpha = 0.5
    scores, idx = search(index_for(design, colour, alpha), qd, qc, alpha, 20)
    expected = alpha * design @ qd + (1 - alpha) * colour @ qc        # the formula on the slides
    assert np.allclose(scores, expected[idx], atol=1e-5)
    assert list(scores) == sorted(scores, reverse=True)               # best first


def test_group_hint_puts_that_group_first(catalogue):
    design, colour, meta = catalogue
    scores, idx = search(index_for(design, colour, 0.5), design[3], colour[3], 0.5, 10,
                         meta=meta, hint="Garment Lower body")
    assert all(meta["group"][j] == "Garment Lower body" for j in idx)  # 10 lower-body items exist
    assert list(scores) == sorted(scores, reverse=True)                # still in score order


def test_final_group_trusts_a_confident_agreeing_tagger():
    assert final_group([("Garment Upper body", 0.9)], "Garment Upper body") == ("Garment Upper body", "90%")


def test_final_group_detector_wins_on_disagreement():
    assert final_group([("Shoes", 0.95)], "Garment Upper body") == ("Garment Upper body", "from detector")


def test_final_group_detector_wins_when_tagger_unsure():
    assert final_group([("Garment Upper body", TAG_TRUST - 0.01)], "Garment Upper body")[1] == "from detector"


def test_final_group_without_hint_uses_tagger():
    assert final_group([("Accessories", 0.4)], None) == ("Accessories", "40%")
