import cv2
import numpy as np
import pytest

from depth_volumetry.io import read_color, read_depth, read_mask, write_image


def test_color_round_trip(tmp_path):
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    image[..., 2] = 200  # red in BGR
    path = tmp_path / "nested" / "color.png"
    write_image(path, image)
    assert np.array_equal(read_color(path), image)


def test_depth_from_npy(tmp_path):
    depth = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.save(tmp_path / "depth.npy", depth)
    assert np.array_equal(read_depth(tmp_path / "depth.npy"), depth)


def test_depth_from_16_bit_png_keeps_values(tmp_path):
    depth = np.array([[0, 1000], [65535, 1234]], dtype=np.uint16)
    cv2.imwrite(str(tmp_path / "depth.png"), depth)
    result = read_depth(tmp_path / "depth.png")
    assert result.dtype == np.uint16
    assert np.array_equal(result, depth)


def test_depth_rejects_multichannel(tmp_path):
    np.save(tmp_path / "depth.npy", np.zeros((2, 2, 3)))
    with pytest.raises(ValueError, match="2-D"):
        read_depth(tmp_path / "depth.npy")


def test_depth_rejects_unknown_extension(tmp_path):
    (tmp_path / "depth.txt").write_text("1 2 3")
    with pytest.raises(ValueError, match="unsupported depth format"):
        read_depth(tmp_path / "depth.txt")


def test_mask_is_boolean(tmp_path):
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    cv2.imwrite(str(tmp_path / "mask.png"), mask)
    result = read_mask(tmp_path / "mask.png")
    assert result.dtype == bool
    assert result.sum() == 4


@pytest.mark.parametrize("reader", [read_color, read_depth, read_mask])
def test_missing_file(reader, tmp_path):
    with pytest.raises(FileNotFoundError, match="file not found"):
        reader(tmp_path / "missing.png")


def test_undecodable_image(tmp_path):
    (tmp_path / "broken.png").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="could not decode"):
        read_color(tmp_path / "broken.png")
