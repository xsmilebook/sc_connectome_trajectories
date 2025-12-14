import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sc_dir_default = "D:\\projects\\sc_connectome_trajectories\\data\\ABCD\\sc_connectome\\schaefer400"
labels_path_default = "D:\\projects\\sc_connectome_trajectories\\data\\atlas\\schaefer400_index.csv"
fig_root = "D:\\projects\\sc_connectome_trajectories\\data\\ABCD\\figures"
fig_dir = os.path.join(fig_root, "cohort")
subject_info_path_default = "D:\\projects\\sc_connectome_trajectories\\data\\ABCD\\table\\subject_info_sc.csv"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def find_first_matrix(sc_dir):
    files = [f for f in os.listdir(sc_dir) if f.endswith('.csv')]
    files.sort()
    return os.path.join(sc_dir, files[0]) if files else None

def load_matrix(path):
    arr = pd.read_csv(path, header=None).values
    if arr.shape[0] >= 400 and arr.shape[1] >= 400:
        arr = arr[:400, :400]
    else:
        n = min(arr.shape[0], arr.shape[1])
        arr = arr[:n, :n]
    return arr.astype(float)

def preprocess_matrix(arr):
    arr = np.array(arr, dtype=float)
    np.fill_diagonal(arr, 0.0)
    arr = np.log1p(arr)
    return arr

def load_labels(labels_path):
    try:
        df = pd.read_csv(labels_path)
    except Exception:
        df = pd.read_csv(labels_path, header=None)
    cols = [c.lower() for c in df.columns] if isinstance(df.columns[0], str) else []
    idx_col = None
    lab_col = None
    if 'index' in cols:
        idx_col = df.columns[cols.index('index')]
    else:
        if hasattr(df, 'columns') and not isinstance(df.columns[0], str):
            idx_col = 1 if df.shape[1] > 1 else 0
    for cname in df.columns:
        cn = str(cname).lower()
        if 'label_7network' in cn or '7network' in cn or 'yeo7' in cn or 'network7' in cn:
            lab_col = cname
            break
    if lab_col is None and hasattr(df, 'columns') and not isinstance(df.columns[0], str):
        lab_col = 3 if df.shape[1] > 3 else 0
    df2 = pd.DataFrame({'index': pd.to_numeric(df[idx_col], errors='coerce') if idx_col is not None else pd.Series(range(len(df))),
                        'label_7network': df[lab_col].astype(str) if lab_col is not None else pd.Series(['']*len(df))})
    df2 = df2.dropna(subset=['index']).copy()
    df2['index'] = df2['index'].astype(int)
    df2 = df2[df2['index'] <= 400]
    return df2

def reorder_by_network(arr, labels_df):
    labels_df = labels_df.sort_values(['label_7network', 'index'])
    order = labels_df['index'].values - 1
    order = order[(order >= 0) & (order < arr.shape[0])]
    arr_ord = arr[np.ix_(order, order)]
    groups = labels_df['label_7network'].values.tolist()
    boundaries = []
    last = None
    for i, g in enumerate(groups):
        if i == 0:
            last = g
            continue
        if g != last:
            boundaries.append(i)
            last = g
    return arr_ord, order, groups, boundaries

def draw_boundaries(ax, boundaries, n):
    for b in boundaries:
        ax.axhline(b, color='black', linewidth=1.2)
        ax.axvline(b, color='black', linewidth=1.2)
    ax.axhline(0, color='black', linewidth=1.2)
    ax.axvline(0, color='black', linewidth=1.2)
    ax.axhline(n, color='black', linewidth=1.2)
    ax.axvline(n, color='black', linewidth=1.2)

def group_labels_positions(groups):
    pos = []
    names = []
    start = 0
    for i in range(len(groups)):
        if i == len(groups)-1 or groups[i+1] != groups[i]:
            end = i
            pos.append((start + end) / 2.0)
            names.append(groups[i])
            start = i + 1
    return pos, names

def plot_panel(arr1, arr2=None, out_path=None, cmap='YlOrBr'):
    sns.set_theme(style='whitegrid')
    plt.rcParams.update({'font.size': 12})
    arr1r = preprocess_matrix(arr1)
    if arr2 is not None:
        arr2r = preprocess_matrix(arr2)
        diff = arr2r - arr1r
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        a1 = sns.heatmap(arr1r, ax=axes[0], cmap=cmap, cbar=True, square=True, vmin=0, cbar_kws={'shrink': 0.6})
        a2 = sns.heatmap(arr2r, ax=axes[1], cmap=cmap, cbar=True, square=True, vmin=0, cbar_kws={'shrink': 0.6})
        ad = sns.heatmap(diff, ax=axes[2], cmap='vlag', center=0.0, cbar=True, square=True, cbar_kws={'shrink': 0.6})
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['top'].set_linewidth(1.5)
            ax.spines['right'].set_linewidth(1.5)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
            ax.spines['top'].set_color('black')
            ax.spines['right'].set_color('black')
            ax.spines['left'].set_color('black')
            ax.spines['bottom'].set_color('black')
        axes[0].set_title('Baseline')
        axes[1].set_title('Year 2')
        axes[2].set_title('Difference (Y2 - Base)')
        axes[0].collections[0].colorbar.set_label('Log(Streamline Count)')
        axes[1].collections[0].colorbar.set_label('Log(Streamline Count)')
        axes[2].collections[0].colorbar.set_label('Δ Log(Streamline Count)')
    else:
        fig, ax = plt.subplots(1, 1, figsize=(9, 9))
        a = sns.heatmap(arr1r, ax=ax, cmap=cmap, cbar=True, square=True, vmin=0, cbar_kws={'shrink': 0.7})
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.set_title('Structural Connectivity (Log-transformed)')
        a.collections[0].colorbar.set_label('Log(Streamline Count)')
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--matrix1', type=str, default=None)
    ap.add_argument('--matrix2', type=str, default=None)
    ap.add_argument('--subject_info', type=str, default=subject_info_path_default)
    ap.add_argument('--sc_dir', type=str, default=sc_dir_default)
    ap.add_argument('--cmap', type=str, default='YlOrBr')
    ap.add_argument('--out', type=str, default=None)
    return ap.parse_args()

def main():
    ensure_dir(fig_dir)
    args = parse_args()
    m1_path = args.matrix1 or find_first_matrix(args.sc_dir)
    if m1_path is None:
        return
    arr1 = load_matrix(m1_path)
    if args.matrix2:
        arr2 = load_matrix(args.matrix2)
        base = os.path.splitext(os.path.basename(m1_path))[0]
        y2 = os.path.splitext(os.path.basename(args.matrix2))[0]
        out_path = args.out or os.path.join(fig_dir, f"sc_matrix_panel_{base}_vs_{y2}.png")
        plot_panel(arr1, arr2=arr2, out_path=out_path, cmap=args.cmap)
    else:
        base = os.path.splitext(os.path.basename(m1_path))[0]
        out_path = args.out or os.path.join(fig_dir, f"sc_matrix_panel_{base}.png")
        plot_panel(arr1, arr2=None, out_path=out_path, cmap=args.cmap)
    if args.subject_info and os.path.exists(args.subject_info):
        df = pd.read_csv(args.subject_info)
        if 'scanid' in df.columns and 'age' in df.columns:
            df = df[['scanid', 'age']].copy()
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
            bins = [(9.0, 11.0, '9-11'), (11.0, 13.0, '11-13'), (13.0, 15.0, '13-15')]
            for low, high, label in bins:
                if label != '13-15':
                    sel = df[(df['age'] >= low) & (df['age'] < high)]['scanid']
                else:
                    sel = df[(df['age'] >= low) & (df['age'] <= high)]['scanid']
                paths = []
                for sid in sel:
                    p = os.path.join(args.sc_dir, f"{sid}.csv")
                    if os.path.exists(p):
                        paths.append(p)
                if not paths:
                    continue
                acc = None
                n = 0
                for p in paths:
                    a = load_matrix(p)
                    a = preprocess_matrix(a)
                    if acc is None:
                        acc = np.zeros_like(a, dtype=float)
                    acc += a
                    n += 1
                if n == 0:
                    continue
                avg = acc / float(n)
                out_path = os.path.join(fig_dir, f"sc_matrix_group_{label.replace('-', '_')}.png")
                plot_panel(avg, arr2=None, out_path=out_path, cmap=args.cmap)

if __name__ == '__main__':
    main()

