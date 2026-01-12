from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.utils import load_matrix, flatten_upper_triangle


class SCDataset(Dataset):
    def __init__(
        self,
        sequences: List[Tuple[str, List[str]]],
        triu_idx: Tuple[np.ndarray, np.ndarray],
    ) -> None:
        self.sequences = sequences
        self.triu_idx = triu_idx

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        subject_id, paths = self.sequences[idx]
        vectors: List[np.ndarray] = []
        for p in paths:
            mat = load_matrix(p)
            vec = flatten_upper_triangle(mat, self.triu_idx)
            vectors.append(vec)
        seq = np.stack(vectors, axis=0)
        x = seq[:-1]
        y = seq[1:]
        length = x.shape[0]
        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "length": length,
            "subject": subject_id,
        }


def collate_sequences(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    lengths = [b["length"] for b in batch]
    if len(lengths) == 0:
        return {}
    max_len = max(lengths)
    feature_dim = batch[0]["x"].shape[1]
    batch_size = len(batch)
    x = torch.zeros(batch_size, max_len, feature_dim, dtype=torch.float32)
    y = torch.zeros(batch_size, max_len, feature_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.float32)
    subjects: List[str] = []
    for i, b in enumerate(batch):
        L = b["length"]
        x[i, :L] = b["x"]
        y[i, :L] = b["y"]
        mask[i, :L] = 1.0
        subjects.append(b["subject"])
    return {
        "x": x,
        "y": y,
        "mask": mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "subjects": subjects,
    }

