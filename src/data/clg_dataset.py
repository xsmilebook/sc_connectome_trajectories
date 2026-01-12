from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.morphology import (
    build_morph_index,
    infer_roi_metric_order,
    load_morphology_matrix,
)
from src.data.utils import load_matrix, parse_subject_session, list_subject_sequences


class CLGDataset(Dataset):
    def __init__(
        self,
        sc_dir: str,
        morph_root: str,
        subject_info_csv: str,
        max_nodes: int = 400,
        min_length: int = 2,
        require_morph: bool = True,
    ) -> None:
        self.sc_dir = sc_dir
        self.morph_root = morph_root
        self.subject_info_csv = subject_info_csv
        self.max_nodes = max_nodes
        self.min_length = min_length
        self.require_morph = require_morph
        self.subject_info = pd.read_csv(subject_info_csv)
        self.scan_info, self.site_vocab_size = _build_scan_info(self.subject_info)
        self.sex_vocab_size = 3
        self.morph_index = build_morph_index(morph_root)
        self.roi_order, self.metric_order = self._infer_morph_layout()
        if self.require_morph and not self.morph_index:
            raise ValueError(f"No morphology files found under: {morph_root}")
        if self.require_morph and not self.metric_order:
            raise ValueError(f"No morphology metrics detected in files under: {morph_root}")
        self.sequences = self._build_sequences()

    def _infer_morph_layout(self) -> Tuple[List[str], List[str]]:
        for scanid, path in self.morph_index.items():
            df = pd.read_csv(path)
            columns = [c for c in df.columns if c != "SubjectID"]
            return infer_roi_metric_order(columns)
        return [], []

    def _build_sequences(self) -> List[Tuple[str, List[str]]]:
        sequences = list_subject_sequences(self.sc_dir, min_length=self.min_length)
        filtered = []
        for sid, paths in sequences:
            scanids = [_scanid_from_path(p) for p in paths]
            if self.require_morph:
                if not all(scanid in self.morph_index for scanid in scanids):
                    continue
            filtered.append((sid, paths))
        return filtered

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        subject_id, paths = self.sequences[idx]
        a_list: List[np.ndarray] = []
        x_list: List[np.ndarray] = []
        times: List[float] = []
        sex_id = 0
        site_id = 0
        for p in paths:
            scanid = _scanid_from_path(p)
            a = load_matrix(p, max_nodes=self.max_nodes)
            if a.shape[0] < self.max_nodes:
                padded = np.zeros((self.max_nodes, self.max_nodes), dtype=np.float32)
                padded[: a.shape[0], : a.shape[1]] = a
                a = padded
            a_list.append(a)
            morph_path = self.morph_index.get(scanid)
            if morph_path and self.roi_order:
                x = load_morphology_matrix(
                    morph_path,
                    roi_order=self.roi_order,
                    metric_order=self.metric_order,
                    max_nodes=self.max_nodes,
                )
            else:
                x = np.zeros((self.max_nodes, len(self.metric_order)), dtype=np.float32)
            x_list.append(x)
            scan_info = self.scan_info.get(scanid, {})
            age = scan_info.get("age", float(len(times)))
            if not np.isfinite(age):
                age = float(len(times))
            times.append(float(age))
            if sex_id == 0:
                sex_id = int(scan_info.get("sex_id", 0))
            if site_id == 0:
                site_id = int(scan_info.get("site_id", 0))
        a_seq = np.stack(a_list, axis=0)
        x_seq = np.stack(x_list, axis=0)
        times_arr = np.array(times, dtype=np.float32)
        return {
            "a": torch.from_numpy(a_seq),
            "x": torch.from_numpy(x_seq),
            "times": torch.from_numpy(times_arr),
            "length": a_seq.shape[0],
            "sex": sex_id,
            "site": site_id,
            "subject": subject_id,
        }


def collate_clg_sequences(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    lengths = [b["length"] for b in batch]
    if not lengths:
        return {}
    max_len = max(lengths)
    batch_size = len(batch)
    n_nodes = batch[0]["a"].shape[1]
    morph_dim = batch[0]["x"].shape[2]
    a = torch.zeros(batch_size, max_len, n_nodes, n_nodes, dtype=torch.float32)
    x = torch.zeros(batch_size, max_len, n_nodes, morph_dim, dtype=torch.float32)
    times = torch.zeros(batch_size, max_len, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.float32)
    sex = torch.zeros(batch_size, dtype=torch.long)
    site = torch.zeros(batch_size, dtype=torch.long)
    subjects: List[str] = []
    for i, b in enumerate(batch):
        L = b["length"]
        a[i, :L] = b["a"]
        x[i, :L] = b["x"]
        times[i, :L] = b["times"]
        mask[i, :L] = 1.0
        sex[i] = b["sex"]
        site[i] = b["site"]
        subjects.append(b["subject"])
    return {
        "a": a,
        "x": x,
        "times": times,
        "mask": mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "sex": sex,
        "site": site,
        "subjects": subjects,
    }


def _scanid_from_path(path: str) -> str:
    base = os.path.basename(path)
    sid, ses = parse_subject_session(base)
    return f"{sid}_{ses}" if ses else sid


def _build_scan_info(df: pd.DataFrame) -> Tuple[Dict[str, Dict[str, float]], int]:
    scan_info: Dict[str, Dict[str, float]] = {}
    site_ids = sorted(set(df.get("siteid", pd.Series(dtype=str)).dropna().astype(str)))
    site_map = {site: i + 1 for i, site in enumerate(site_ids)}
    for _, row in df.iterrows():
        scanid = str(row.get("scanid", "")).strip()
        if not scanid:
            continue
        age_raw = row.get("age", "")
        try:
            age = float(age_raw)
        except (TypeError, ValueError):
            age = float("nan")
        sex_val = str(row.get("sex", "")).strip().lower()
        if sex_val in {"m", "male", "1"}:
            sex_id = 1
        elif sex_val in {"f", "female", "2"}:
            sex_id = 2
        else:
            sex_id = 0
        site_val = str(row.get("siteid", "")).strip()
        site_id = site_map.get(site_val, 0)
        scan_info[scanid] = {"age": age, "sex_id": sex_id, "site_id": site_id}
    return scan_info, max(site_map.values(), default=0) + 1
