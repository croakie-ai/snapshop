"""The next-item models: shapes, padding, and no peeking at future purchases."""
import pytest
import torch

from train_recommender import DIM, Recommender, pad

N_ITEMS = 50


def test_pad_right_pads_with_zeros():
    x = pad([[5, 6, 7], [9]])
    assert x.tolist() == [[5, 6, 7], [9, 0, 0]]
    assert x.dtype == torch.long


@pytest.mark.parametrize("kind", ["lstm", "transformer"])
def test_output_shapes(kind):
    torch.manual_seed(0)
    model = Recommender(N_ITEMS, kind).eval()
    x = pad([[1, 2, 3, 4], [5, 6]])
    h = model(x)
    assert h.shape == (2, 4, DIM)                         # a summary at every position
    assert model.scores(h[:, -1]).shape == (2, N_ITEMS)   # one score per product (weight tying)


@pytest.mark.parametrize("kind", ["lstm", "transformer"])
def test_no_peeking_at_future_items(kind):
    """Changing a LATER purchase must not change the prediction made at an EARLIER position
    (the transformer's causal mask; the LSTM reads left to right)."""
    torch.manual_seed(0)
    model = Recommender(N_ITEMS, kind).eval()
    with torch.no_grad():
        a = model(torch.tensor([[1, 2, 3, 4]]))
        b = model(torch.tensor([[1, 2, 3, 40]]))           # only the last item differs
    assert torch.allclose(a[0, :3], b[0, :3], atol=1e-6)
    assert not torch.allclose(a[0, 3], b[0, 3])
