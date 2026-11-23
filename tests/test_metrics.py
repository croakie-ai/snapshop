"""MAP@12 (the recommender's metric) on hand-computed examples."""
import pytest

from train_recommender import apk


def test_apk_perfect_list_scores_one():
    assert apk(["a", "b"], ["a", "b", "x"]) == 1.0


def test_apk_no_hits_scores_zero():
    assert apk(["a"], ["x", "y", "z"]) == 0.0


def test_apk_hand_computed_example():
    # hits at rank 1 and rank 3: precision 1/1 and 2/3, averaged over 2 bought items
    assert apk(["a", "b"], ["a", "x", "b"]) == pytest.approx((1 / 1 + 2 / 3) / 2)


def test_apk_rewards_hits_near_the_top():
    assert apk(["a"], ["a", "x", "y"]) > apk(["a"], ["x", "y", "a"])


def test_apk_ignores_items_after_k():
    assert apk(["a"], ["x"] * 12 + ["a"], k=12) == 0.0


def test_apk_counts_a_repeated_prediction_once():
    assert apk(["a", "b"], ["a", "a", "b"]) == pytest.approx((1 / 1 + 2 / 3) / 2)


def test_apk_empty_basket_scores_zero():
    assert apk([], ["a", "b"]) == 0.0
