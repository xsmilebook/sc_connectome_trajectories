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
from src.data.topology import compute_ecc
from src.data.utils import (
    compute_strengths,
    load_matrix,
    list_subject_sequences,
    parse_subject_session,
    preprocess_sc,
)


class CLGDataset(Dataset):
    def __init__(
        self,
        sc_dir: str,
        morph_root: str,
        subject_info_csv: str,
        max_nodes: int = 400,
        min_length: int = 2,
        require_morph: bool = True,
        topo_bins: int = 32,
    ) -> None:
        self.sc_dir = sc_dir
        self.morph_root = morph_root
        self.subject_info_csv = subject_info_csv
        self.max_nodes = max_nodes
        self.min_length = min_length
        self.require_morph = require_morph
        self.topo_bins = topo_bins
        self.subject_info = pd.read_csv(subject_info_csv)
        self.scan_info, self.site_vocab_size = _build_scan_info(self.subject_info)
        self.sex_vocab_size = 3
        self.morph_index = build_morph_index(morph_root)
        self.roi_order, self.metric_order = self._infer_morph_layout()
        self.volume_metric_indices = [
            i for i, m in enumerate(self.metric_order) if "vol" in m.lower()
        ]
        self._sc_cache: Dict[str, Dict[str, Any]] = {}
        self._morph_cache: Dict[str, np.ndarray] = {}
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
        a_raw_list: List[np.ndarray] = []
        a_log_list: List[np.ndarray] = []
        x_list: List[np.ndarray] = []
        ages: List[float] = []
        topo_list: List[np.ndarray] = []
        strength_list: List[np.ndarray] = []
        icv_list: List[float] = []
        sex_id = 0
        site_id = 0
        for p in paths:
            scanid = _scanid_from_path(p)
            cached = self._sc_cache.get(p)
            if cached is None:
                a = load_matrix(p, max_nodes=self.max_nodes)
                if a.shape[0] < self.max_nodes:
                    padded = np.zeros((self.max_nodes, self.max_nodes), dtype=np.float32)
                    padded[: a.shape[0], : a.shape[1]] = a
                    a = padded
                a_raw, a_log = preprocess_sc(a)
                s, s_mean = compute_strengths(a_raw)
                topo = compute_ecc(a_log, k=self.topo_bins)
                cached = {
                    "a_raw": a_raw,
                    "a_log": a_log,
                    "strength": np.array([s, s_mean], dtype=np.float32),
                    "topo": topo.astype(np.float32),
                }
                self._sc_cache[p] = cached
            a_raw_list.append(cached["a_raw"])
            a_log_list.append(cached["a_log"])
            topo_list.append(cached["topo"])
            strength_list.append(cached["strength"])
            morph_path = self.morph_index.get(scanid)
            if morph_path and self.roi_order:
                cached_x = self._morph_cache.get(morph_path)
                if cached_x is None:
                    cached_x = load_morphology_matrix(
                        morph_path,
                        roi_order=self.roi_order,
                        metric_order=self.metric_order,
                        max_nodes=self.max_nodes,
                    )
                    self._morph_cache[morph_path] = cached_x
                x = cached_x
            else:
                x = np.zeros((self.max_nodes, len(self.metric_order)), dtype=np.float32)
            x_list.append(x)
            scan_info = self.scan_info.get(scanid, {})
            age = scan_info.get("age", float(len(ages)))
            if not np.isfinite(age):
                age = float(len(ages))
            ages.append(float(age))
            icv_list.append(float(scan_info.get("icv", float("nan"))))
            if sex_id == 0:
                sex_id = int(scan_info.get("sex_id", 0))
            if site_id == 0:
                site_id = int(scan_info.get("site_id", 0))
        a_raw_seq = np.stack(a_raw_list, axis=0)
        a_log_seq = np.stack(a_log_list, axis=0)
        x_seq = np.stack(x_list, axis=0)
        ages_arr = np.array(ages, dtype=np.float32)
        topo_arr = np.stack(topo_list, axis=0)
        strength_arr = np.stack(strength_list, axis=0)
        icv_arr = np.array(icv_list, dtype=np.float32)
        return {
            "a_raw": torch.from_numpy(a_raw_seq),
            "a_log": torch.from_numpy(a_log_seq),
            "x": torch.from_numpy(x_seq),
            "ages": torch.from_numpy(ages_arr),
            "topo": torch.from_numpy(topo_arr),
            "strength": torch.from_numpy(strength_arr),
            "icv": torch.from_numpy(icv_arr),
            "length": a_raw_seq.shape[0],
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
    n_nodes = batch[0]["a_raw"].shape[1]
    morph_dim = batch[0]["x"].shape[2]
    topo_dim = batch[0]["topo"].shape[1]
    a_raw = torch.zeros(batch_size, max_len, n_nodes, n_nodes, dtype=torch.float32)
    a_log = torch.zeros(batch_size, max_len, n_nodes, n_nodes, dtype=torch.float32)
    x = torch.zeros(batch_size, max_len, n_nodes, morph_dim, dtype=torch.float32)
    ages = torch.zeros(batch_size, max_len, dtype=torch.float32)
    topo = torch.zeros(batch_size, max_len, topo_dim, dtype=torch.float32)
    strength = torch.zeros(batch_size, max_len, 2, dtype=torch.float32)
    icv = torch.zeros(batch_size, max_len, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.float32)
    sex = torch.zeros(batch_size, dtype=torch.long)
    site = torch.zeros(batch_size, dtype=torch.long)
    subjects: List[str] = []
    for i, b in enumerate(batch):
        L = b["length"]
        a_raw[i, :L] = b["a_raw"]
        a_log[i, :L] = b["a_log"]
        x[i, :L] = b["x"]
        ages[i, :L] = b["ages"]
        topo[i, :L] = b["topo"]
        strength[i, :L] = b["strength"]
        icv[i, :L] = b["icv"]
        mask[i, :L] = 1.0
        sex[i] = b["sex"]
        site[i] = b["site"]
        subjects.append(b["subject"])
    return {
        "a_raw": a_raw,
        "a_log": a_log,
        "x": x,
        "ages": ages,
        "topo": topo,
        "strength": strength,
        "icv": icv,
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
    cols_lower = {c.lower(): c for c in df.columns}
    icv_key = cols_lower.get("icv")
    tiv_key = cols_lower.get("tiv")
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
        icv_val = float("nan")
        if icv_key:
            try:
                icv_val = float(row.get(icv_key, "nan"))
            except (TypeError, ValueError):
                icv_val = float("nan")
        elif tiv_key:
            try:
                icv_val = float(row.get(tiv_key, "nan"))
            except (TypeError, ValueError):
                icv_val = float("nan")
        scan_info[scanid] = {
            "age": age,
            "sex_id": sex_id,
            "site_id": site_id,
            "icv": icv_val,
        }
    return scan_info, max(site_map.values(), default=0) + 1
