import os
import csv
from collections import Counter

sc_folder = "D:\\projects\\sc_connectome_trajectories\\data\\ABCD\\sc_connectome\\schaefer400"
input_csv = "D:\\projects\\sc_connectome_trajectories\\data\\ABCD\\table\\abcd_y_lt.csv"
output_csv = "D:\\projects\\sc_connectome_trajectories\\data\\ABCD\\table\\subject_info_sc.csv"

event_map = {
    "2_year_follow_up_y_arm_1": "ses-2YearFollowUpYArm1",
    "4_year_follow_up_y_arm_1": "ses-4YearFollowUpYArm1",
    "baseline_year_1_arm_1": "ses-baselineYear1Arm1",
}

def list_scans(folder):
    scans = set()
    for name in os.listdir(folder):
        if not name.endswith(".csv"):
            continue
        base = os.path.splitext(name)[0]
        if base.startswith("sub-") and "_ses-" in base:
            scans.add(base)
    return scans

def build_sex_map(input_path):
    sex_by_subid = {}
    with open(input_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_sid = row.get("src_subject_id", "")
            if not raw_sid:
                continue
            sid = ("sub-" + raw_sid.replace("_", "")).strip()
            sex_val = row.get("demo_sex_v2", "").strip()
            if sex_val and sid not in sex_by_subid:
                sex_by_subid[sid] = sex_val
    return sex_by_subid

def build_subject_rows(input_path, scans, sex_by_subid):
    rows = []
    seen = set()
    with open(input_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = ("sub-" + row.get("src_subject_id", "").replace("_", "")).strip()
            ev = row.get("eventname", "").strip()
            ses = event_map.get(ev, "")
            if not ses:
                continue
            scanid = sid + "_" + ses
            if scanid not in scans:
                continue
            if scanid in seen:
                continue
            siteid = row.get("site_id_l", "").strip()
            age = row.get("interview_age", "").strip()
            sex_here = row.get("demo_sex_v2", "").strip()
            sex_val = sex_here if sex_here else sex_by_subid.get(sid, "")
            rows.append({
                "subid": sid,
                "sesid": ses,
                "scanid": scanid,
                "siteid": siteid,
                "age": age,
                "sex": sex_val,
            })
            seen.add(scanid)
    return rows

def write_csv(path, rows):
    fieldnames = ["subid", "sesid", "scanid", "siteid", "age", "sex"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def summarize(rows, scans):
    print("SC文件数量:", len(scans))
    print("有效被试数量:", len(rows))
    ses_counts = Counter(r["sesid"] for r in rows)
    site_counts = Counter(r["siteid"] for r in rows)
    sex_counts = Counter(r["sex"] for r in rows)
    print("分期分布:")
    for k, v in ses_counts.items():
        print(k, v)
    print("站点分布:")
    for k, v in site_counts.items():
        print(k, v)
    print("性别分布:")
    for k, v in sex_counts.items():
        print(k, v)

def main():
    scans = list_scans(sc_folder)
    sex_by_subid = build_sex_map(input_csv)
    rows = build_subject_rows(input_csv, scans, sex_by_subid)
    write_csv(output_csv, rows)
    summarize(rows, scans)

if __name__ == "__main__":
    main()

