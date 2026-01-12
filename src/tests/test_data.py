from unittest.mock import patch

import numpy as np

from data.dataset import SCDataset, collate_sequences


def test_sc_dataset_sequence_shapes():
    sequences = [
        ("sub-001", ["file1.csv", "file2.csv", "file3.csv"]),
    ]
    triu_idx = np.triu_indices(4, k=1)

    def fake_load_matrix(path, max_nodes=400):
        return np.ones((4, 4), dtype=np.float32)

    with patch("data.dataset.load_matrix", side_effect=fake_load_matrix):
        ds = SCDataset(sequences, triu_idx)
        item = ds[0]
        assert item["x"].shape[0] == 2
        assert item["y"].shape[0] == 2
        assert item["x"].shape[1] == triu_idx[0].shape[0]
        batch = collate_sequences([item])
        assert batch["x"].shape[0] == 1
        assert batch["x"].shape[1] == 2
        assert batch["x"].shape[2] == triu_idx[0].shape[0]

