import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

input_csv = "D:\\projects\\sc_connectome_trajectories\\data\\ABCD\\table\\subject_info_sc.csv"
fig_root = "D:\\projects\\sc_connectome_trajectories\\data\\ABCD\\figures"
fig_dir = os.path.join(fig_root, "cohort")

ses_map = {
    "ses-baselineYear1Arm1": "Baseline",
    "ses-2YearFollowUpYArm1": "Year 2",
    "ses-4YearFollowUpYArm1": "Year 4",
}

def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["subid", "sesid"]) 
    df["ses_simple"] = df["sesid"].map(ses_map)
    df = df[df["ses_simple"].notna()]
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    if df["age"].max() is not None and df["age"].max() > 25:
        df["age_years"] = (df["age"] / 12).round(1)
    else:
        df["age_years"] = df["age"].round(1)
    df["sex"] = df["sex"].astype(str)
    df["sex_label"] = df["sex"].map({"1": "Male", "2": "Female"}).fillna(df["sex"]) 
    return df

def retention_counts(df):
    g = df.groupby(["ses_simple"]) ["subid"].nunique()
    return g.reindex(["Baseline", "Year 2", "Year 4"]).fillna(0)

def completeness_counts(df):
    ses_per_sub = df.groupby("subid")["ses_simple"].nunique()
    counts = ses_per_sub.value_counts()
    return pd.Series({1: counts.get(1, 0), 2: counts.get(2, 0), 3: counts.get(3, 0)})

def site_subject_counts(df):
    sub_site = df.sort_values(["subid", "siteid"]).groupby("subid").first().reset_index()[["subid", "siteid", "sex_label"]]
    c = sub_site.groupby(["siteid", "sex_label"]).size().reset_index(name="count")
    totals = c.groupby("siteid")["count"].sum().sort_values(ascending=False)
    c["siteid"] = pd.Categorical(c["siteid"], categories=totals.index.tolist(), ordered=True)
    c = c.sort_values(["siteid", "sex_label"]) 
    return c, totals

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def plot_all(df, save_dir):
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.size": 12})
    palette = sns.color_palette("Set2")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ret = retention_counts(df)
    ax = axes[0, 0]
    sns.barplot(x=ret.index, y=ret.values, ax=ax, palette=palette)
    for i, v in enumerate(ret.values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom")
    ax.set_title("Longitudinal Retention")
    ax.set_xlabel("Session")
    ax.set_ylabel("Unique Subjects")

    comp = completeness_counts(df)
    ax = axes[0, 1]
    sns.barplot(x=comp.index.astype(str), y=comp.values, ax=ax, palette=palette)
    for i, v in enumerate(comp.values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom")
    ax.set_title("Data Completeness")
    ax.set_xlabel("# Sessions Available")
    ax.set_ylabel("Subjects")

    ax = axes[1, 0]
    sns.violinplot(data=df, x="ses_simple", y="age_years", ax=ax, inner="quartile", palette=palette)
    sns.stripplot(data=df, x="ses_simple", y="age_years", ax=ax, color="k", alpha=0.15, jitter=True)
    ax.set_title("Age Distribution by Session")
    ax.set_xlabel("Session")
    ax.set_ylabel("Age (years)")

    ax = axes[1, 1]
    site_counts, totals = site_subject_counts(df)
    cats = site_counts["sex_label"].unique().tolist()
    bottom = np.zeros(len(totals))
    for i, cat in enumerate(cats):
        vals = site_counts[site_counts["sex_label"] == cat].set_index("siteid")["count"].reindex(totals.index, fill_value=0).values
        ax.bar(totals.index, vals, bottom=bottom, label=cat, color=palette[i % len(palette)])
        bottom += vals
    ax.set_title("Site Heterogeneity (Stacked by Sex)")
    ax.set_xlabel("Site ID")
    ax.set_ylabel("Unique Subjects")
    ax.legend(title="Sex")
    for label in ax.get_xticklabels():
        label.set_rotation(45)

    plt.tight_layout()
    ensure_dir(save_dir)
    out_path = os.path.join(save_dir, "cohort_overview.png")
    fig.savefig(out_path, dpi=300)

def main():
    ensure_dir(fig_dir)
    df = load_and_clean(input_csv)
    plot_all(df, fig_dir)

if __name__ == "__main__":
    main()

