"""Preprocessing must match training exactly (train/serve skew is a classic production bug)."""
import torch
from PIL import Image

from data.images import IMAGENET_MEAN, IMAGENET_STD
from data.make_images import pad_to_square
from predict import prepare


def test_pad_to_square_wide_image_is_centred_on_white():
    img = Image.new("RGB", (300, 100), (255, 0, 0))            # red, wider than tall
    sq = pad_to_square(img)
    assert sq.size == (300, 300)
    assert sq.getpixel((150, 150)) == (255, 0, 0)               # centre = the original photo
    assert sq.getpixel((0, 0)) == (255, 255, 255)               # corners = white padding
    assert sq.getpixel((150, 99)) == (255, 255, 255)            # above the photo (rows 0-99) = white
    assert sq.getpixel((150, 100)) == (255, 0, 0)               # photo starts at row 100 = (300-100)/2


def test_pad_to_square_keeps_square_images():
    img = Image.new("RGB", (64, 64), (0, 0, 255))
    assert pad_to_square(img).size == (64, 64)


def test_prepare_gives_one_normalised_224_batch():
    x = prepare(Image.new("RGB", (500, 300), (255, 255, 255)))  # any size in
    assert x.shape == (1, 3, 224, 224)                          # batch of one, 224 x 224 out
    assert x.dtype == torch.float32
    # an all-white photo -> every pixel equals (1 - mean) / std for its channel
    for c in range(3):
        expected = (1 - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        assert torch.allclose(x[0, c], torch.full((224, 224), expected), atol=1e-4)
