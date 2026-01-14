import numpy as np

from src.data.topology import compute_ecc
from src.data.utils import compute_strengths, preprocess_sc


def test_preprocess_sc_symmetry_and_log():
    mat = np.array([[0.0, 1.0], [3.0, 0.0]], dtype=np.float32)
    raw, log_mat = preprocess_sc(mat)
    assert np.allclose(raw, np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.float32))
    assert np.allclose(log_mat, np.log1p(raw))


def test_compute_strengths():
    mat = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    s, s_mean = compute_strengths(mat, eps=1e-8)
    expected = float(np.log(1.0 + 1e-8))
    assert abs(s - expected) < 1e-6
    assert abs(s_mean - expected) < 1e-6


def test_compute_ecc_shape():
    mat = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    ecc = compute_ecc(np.log1p(mat), k=32)
    assert ecc.shape == (32,)
    assert np.isfinite(ecc).all()
