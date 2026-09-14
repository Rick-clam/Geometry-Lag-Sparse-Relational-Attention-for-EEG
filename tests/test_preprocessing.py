import numpy as np

from src.data.preprocessing import preprocess_record


def test_preprocessing_shape_and_finiteness():
    rng = np.random.default_rng(7)
    signal = rng.standard_normal((4, 30 * 256), dtype=np.float32)
    windows = preprocess_record(signal, 256.0, ["F3", "F4", "C3", "C4"])
    assert windows.shape == (1, 4, 30, 200)
    assert windows.dtype == np.float32
    assert np.isfinite(windows).all()

