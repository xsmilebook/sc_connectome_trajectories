from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

from src.configs.paths import get_by_dotted_key, load_simple_yaml, resolve_repo_path


def infer_session_id(text: str) -> str:
    patterns = [
        ("ses-baselineYear1Arm1", "ses-baselineYear1Arm1"),
        ("baselineYear1Arm1", "ses-baselineYear1Arm1"),
        ("BaselineYear1Arm1", "ses-baselineYear1Arm1"),
        ("baseline", "ses-baselineYear1Arm1"),
        ("ses-2YearFollowUpYArm1", "ses-2YearFollowUpYArm1"),
        ("2YearFollowUpYArm1", "ses-2YearFollowUpYArm1"),
        ("2_year_follow_up_y_arm_1", "ses-2YearFollowUpYArm1"),
        ("2year", "ses-2YearFollowUpYArm1"),
        ("ses-4YearFollowUpYArm1", "ses-4YearFollowUpYArm1"),
        ("4YearFollowUpYArm1", "ses-4YearFollowUpYArm1"),
        ("4_year_follow_up_y_arm_1", "ses-4YearFollowUpYArm1"),
        ("4year", "ses-4YearFollowUpYArm1"),
        ("ses-6YearFollowUpYArm1", "ses-6YearFollowUpYArm1"),
        ("6YearFollowUpYArm1", "ses-6YearFollowUpYArm1"),
        ("6_year_follow_up_y_arm_1", "ses-6YearFollowUpYArm1"),
        ("6year", "ses-6YearFollowUpYArm1"),
    ]
    for token, sesid in patterns:
        if token in text:
            return sesid
    return ""


def _normalize_subid(raw: str) -> str:
    s = raw.strip()
    if not s:
        return ""
    if s.startswith("sub-"):
        return s
    if re.fullmatch(r"NDARINV[A-Za-z0-9]+", s):
        return "sub-" + s
    return s


def parse_subject_session_from_name(filename: str) -> Tuple[str, str]:
    base = os.path.splitext(os.path.basename(filename))[0]
    if "_ses-" in base:
        parts = base.split("_ses-", 1)
        subject_id = _normalize_subid(parts[0])
        session_id = "ses-" + parts[1]
        return subject_id, session_id
    subject_id = _normalize_subid(base.split("_", 1)[0])
    if not subject_id:
        subject_id = _normalize_subid(base)
    m = re.search(r"(sub-[A-Za-z0-9]+)", base)
    if m:
        subject_id = m.group(1)
    session_id = infer_session_id(base)
    return subject_id, session_id


def scanid_from_sc_filename(fname: str) -> str:
    base = os.path.splitext(os.path.basename(fname))[0]
    if "_ses-" in base:
        return base
    sid, ses = parse_subject_session_from_name(fname)
    if sid and ses:
        return f"{sid}_{ses}"
    return ""


def scanid_from_morph_filename(path: str) -> str:
    fname = os.path.basename(path)
    base = os.path.splitext(fname)[0]
    if not base.startswith("Schaefer400_Morphology_"):
        return ""
    base = base[len("Schaefer400_Morphology_") :]
    subid = _normalize_subid(base)
    if not subid.startswith("sub-"):
        return ""
    sesid = infer_session_id(path)
    if not sesid:
        return ""
    return f"{subid}_{sesid}"


def index_sc(sc_dir: str) -> Tuple[Dict[str, str], Counter]:
    errors: Counter = Counter()
    out: Dict[str, str] = {}
    if not os.path.isdir(sc_dir):
        errors["missing_sc_dir"] += 1
        return out, errors
    for fname in os.listdir(sc_dir):
        if not fname.endswith(".csv"):
            continue
        scanid = scanid_from_sc_filename(fname)
        if not scanid:
            errors["sc_parse_failed"] += 1
            continue
        out[scanid] = os.path.join(sc_dir, fname)
    return out, errors


def index_morph(morph_root: str) -> Tuple[Dict[str, str], Counter]:
    errors: Counter = Counter()
    out: Dict[str, str] = {}
    if not os.path.isdir(morph_root):
        errors["missing_morph_root"] += 1
        return out, errors
    for root, _, files in os.walk(morph_root):
        for fname in files:
            if not (fname.startswith("Schaefer400_Morphology_") and fname.endswith(".csv")):
                continue
            p = os.path.join(root, fname)
            scanid = scanid_from_morph_filename(p)
            if not scanid:
                errors["morph_parse_failed"] += 1
                continue
            out.setdefault(scanid, p)
    return out, errors


def split_scanid(scanid: str) -> Tuple[str, str]:
    if "_ses-" not in scanid:
        return scanid, ""
    sid, rest = scanid.split("_ses-", 1)
    return sid, "ses-" + rest


@dataclass(frozen=True)
class TierSummary:
    n_subjects: int
    subjects_by_sessions: Dict[int, int]
    tier1_subjects: int
    tier2_subjects: int
    tier3_subjects: int
    n_sc_sessions: int
    n_morph_sessions: int
    n_joint_sessions: int
    drop_reasons: Dict[str, int]


def compute_tiers(sc_index: Dict[str, str], morph_index: Dict[str, str]) -> Tuple[TierSummary, Dict[str, List[str]]]:
    drop = Counter()
    for scanid, p in sc_index.items():
        if not os.path.exists(p):
            drop["sc_missing_file"] += 1
    for scanid, p in morph_index.items():
        if not os.path.exists(p):
            drop["morph_missing_file"] += 1

    joint_scanids = sorted(set(sc_index) & set(morph_index))
    only_sc = sorted(set(sc_index) - set(morph_index))
    only_morph = sorted(set(morph_index) - set(sc_index))
    if only_sc:
        drop["missing_morph_for_sc"] += len(only_sc)
    if only_morph:
        drop["missing_sc_for_morph"] += len(only_morph)

    subject_sessions: Dict[str, List[str]] = defaultdict(list)
    for scanid in joint_scanids:
        sid, ses = split_scanid(scanid)
        if not sid:
            drop["scanid_missing_subject"] += 1
            continue
        if not ses:
            drop["scanid_missing_session"] += 1
            continue
        subject_sessions[sid].append(ses)

    subjects_by_sessions = Counter({sid: len(sorted(set(sessions))) for sid, sessions in subject_sessions.items()})
    dist = Counter(subjects_by_sessions.values())

    tier1 = sum(1 for n in subjects_by_sessions.values() if n >= 3)
    tier2 = sum(1 for n in subjects_by_sessions.values() if n == 2)
    tier3 = sum(1 for n in subjects_by_sessions.values() if n == 1)

    summary = TierSummary(
        n_subjects=len(subject_sessions),
        subjects_by_sessions=dict(sorted(dist.items())),
        tier1_subjects=tier1,
        tier2_subjects=tier2,
        tier3_subjects=tier3,
        n_sc_sessions=len(sc_index),
        n_morph_sessions=len(morph_index),
        n_joint_sessions=len(joint_scanids),
        drop_reasons=dict(drop),
    )
    details = {sid: sorted(set(sessions)) for sid, sessions in subject_sessions.items()}
    return summary, details


def write_json(path: str, payload: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_subject_csv(path: str, details: Dict[str, List[str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subject_id", "n_sessions", "sessions"])
        for sid, sessions in sorted(details.items()):
            writer.writerow([sid, len(sessions), ";".join(sessions)])


def write_report_md(path: str, summary: TierSummary, sc_dir: str, morph_root: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []
    lines.append("# CLG-ODE 训练数据分层统计（严格文件存在性）")
    lines.append("")
    lines.append(f"- 生成时间：{now}")
    lines.append(f"- SC 目录：`{sc_dir}`")
    lines.append(f"- Morphology 目录：`{morph_root}`")
    lines.append("")
    lines.append("## 统计口径")
    lines.append("- 以 `sc_dir` 中的 SC CSV 文件为起点，按文件名解析 `scanid=sub-..._ses-...`。")
    lines.append("- 以 `morph_root` 下的 `Schaefer400_Morphology_sub-*.csv` 为 morphology 输入，按路径解析 session。")
    lines.append("- **严格要求**：同一 `scanid` 的 SC 与 morphology 文件同时存在，才计入可训练访视。")
    lines.append("")
    lines.append("## 结果汇总")
    lines.append(f"- SC 可用访视数：{summary.n_sc_sessions}")
    lines.append(f"- Morphology 可用访视数：{summary.n_morph_sessions}")
    lines.append(f"- SC∩Morphology 可用访视数：{summary.n_joint_sessions}")
    lines.append(f"- 可训练被试数（至少 1 个访视）：{summary.n_subjects}")
    lines.append("")
    lines.append("## 按被试可用 session 数分布")
    for n_ses, n_sub in summary.subjects_by_sessions.items():
        lines.append(f"- {n_ses} 个 session：{n_sub} 名被试")
    lines.append("")
    lines.append("## Tier 定义与数量")
    lines.append("- Tier 1（≥3 个时间点）：用于学习加速度/非线性项（`L_acc`）。")
    lines.append("- Tier 2（2 个时间点）：用于学习速度场（`L_vel`）。")
    lines.append("- Tier 3（1 个时间点）：用于学习群体流形分布/去噪（`L_manifold`）。")
    lines.append("")
    lines.append(f"- Tier 1：{summary.tier1_subjects}")
    lines.append(f"- Tier 2：{summary.tier2_subjects}")
    lines.append(f"- Tier 3：{summary.tier3_subjects}")
    lines.append("")
    if summary.drop_reasons:
        lines.append("## 丢弃原因（如有）")
        for k, v in sorted(summary.drop_reasons.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"- {k}：{v}")
        lines.append("")
    lines.append("## 产出文件")
    base = os.path.splitext(os.path.basename(path))[0]
    lines.append(f"- 详细被试列表：`docs/reports/{base}_subjects.csv`")
    lines.append(f"- 机器可读摘要：`docs/reports/{base}.json`")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/paths.yaml")
    p.add_argument("--sc_dir", default="")
    p.add_argument("--morph_root", default="")
    p.add_argument("--out_dir", default="docs/reports")
    p.add_argument("--out_prefix", default="clg_ode_dataset_tiers")
    args = p.parse_args(list(argv) if argv is not None else None)

    cfg = load_simple_yaml(resolve_repo_path(args.config))
    sc_dir = args.sc_dir or resolve_repo_path(get_by_dotted_key(cfg, "local.data.sc_connectome_schaefer400"))
    morph_root = args.morph_root or resolve_repo_path(get_by_dotted_key(cfg, "local.data.morphology"))

    sc_index, sc_err = index_sc(sc_dir)
    morph_index, morph_err = index_morph(morph_root)
    summary, details = compute_tiers(sc_index, morph_index)
    summary_payload = {
        "summary": summary.__dict__,
        "index_errors": {**dict(sc_err), **dict(morph_err)},
    }

    out_base = os.path.join(args.out_dir, args.out_prefix)
    write_json(out_base + ".json", summary_payload)
    write_subject_csv(out_base + "_subjects.csv", details)
    write_report_md(out_base + ".md", summary, sc_dir=sc_dir, morph_root=morph_root)

    print(json.dumps(summary_payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

