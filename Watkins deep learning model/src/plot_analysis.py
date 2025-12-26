import argparse
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import torch
except Exception:
    torch = None


def ensure_outdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def plot_learning_curves(hist_paths, labels, outdir: Path):
    ensure_outdir(outdir)
    # Use a seaborn/matplotlib style that's available across versions.
    # Newer matplotlib may not accept 'seaborn-darkgrid', prefer a robust choice.
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except Exception:
        try:
            plt.style.use('seaborn-darkgrid')
        except Exception:
            plt.style.use('seaborn')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for p, label in zip(hist_paths, labels):
        df = pd.read_csv(p)
        axes[0].plot(df['epoch'], df['train_acc'], label=f'{label} train')
        axes[0].plot(df['epoch'], df['val_acc'], '--', label=f'{label} val')
        axes[1].plot(df['epoch'], df['train_loss'], label=f'{label} train')
        axes[1].plot(df['epoch'], df['val_loss'], '--', label=f'{label} val')

    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('accuracy (%)')
    axes[0].legend()
    axes[0].set_title('Accuracy Learning Curve')

    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('loss')
    axes[1].legend()
    axes[1].set_title('Loss Learning Curve')

    fig.tight_layout()
    out = outdir / 'learning_curves.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print('Saved', out)


def plot_precision_recall_from_report(report_path: Path, outdir: Path, top_n: int = 20):
    ensure_outdir(outdir)
    with open(report_path, 'r') as fh:
        report = json.load(fh)

    # filter out overall metrics
    per_class = {k: v for k, v in report.items() if isinstance(v, dict)}
    df = pd.DataFrame(per_class).T
    df['precision'] = df['precision'].astype(float)
    df['recall'] = df['recall'].astype(float)
    df['support'] = df['support'].astype(float)
    df = df.sort_values('support', ascending=False)
    df_sel = df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(1, 2, figsize=(14, max(6, top_n * 0.3 + 2)))
    df_sel['precision'].plot.barh(ax=ax[0], color='C0')
    ax[0].set_title('Precision (top by support)')
    ax[0].set_xlabel('precision')

    df_sel['recall'].plot.barh(ax=ax[1], color='C1')
    ax[1].set_title('Recall (top by support)')
    ax[1].set_xlabel('recall')

    fig.tight_layout()
    out = outdir / (report_path.stem + '_precision_recall.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print('Saved', out)


def plot_strip_predictions(preds_path: Path, test_csv: Path, outdir: Path, top_n_species: int = 12, max_per_species: int = 200):
    ensure_outdir(outdir)
    preds = pd.read_csv(preds_path)
    test = pd.read_csv(test_csv)
    if len(preds) == len(test):
        preds = preds.copy()
        preds['filename'] = test['filename'].values
    else:
        print('Warning: preds and test lengths differ; continuing without filenames')

    preds['correct'] = preds['y_true_name'] == preds['y_pred_name']

    # choose top species by frequency in test set
    top_species = test['category'].value_counts().head(top_n_species).index.tolist()
    df_plot = preds[preds['y_true_name'].isin(top_species)].copy()

    # subsample per species to keep plot readable
    df_list = []
    for s in top_species:
        sub = df_plot[df_plot['y_true_name'] == s]
        if len(sub) > max_per_species:
            sub = sub.sample(max_per_species, random_state=42)
        df_list.append(sub)
    df_plot = pd.concat(df_list)

    plt.figure(figsize=(12, max(6, len(top_species) * 0.4)))
    sns.stripplot(x='y_prob_max', y='y_true_name', data=df_plot, hue='correct', dodge=False, alpha=0.6, jitter=0.3, palette={True: 'C2', False: 'C3'})
    plt.xlabel('predicted probability (max)')
    plt.ylabel('true species')
    plt.title('Strip plot of predictions by confidence (true vs false)')
    plt.legend(title='correct')
    out = outdir / (preds_path.stem + '_stripplot.png')
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print('Saved', out)


def load_spectrogram_tensor(path: Path):
    if not path.exists():
        return None
    if torch is None:
        print('torch not available; cannot load .pt spectrograms')
        return None
    try:
        t = torch.load(str(path), map_location='cpu')
    except Exception as e:
        print('error loading', path, e)
        return None

    # t might be a tensor or a dict with key 'spec' or 'mel' or similar
    if hasattr(t, 'numpy'):
        arr = t.numpy()
    elif isinstance(t, dict):
        # try common keys
        for k in ('spec', 'mel', 'spectrogram', 'S'):
            if k in t:
                v = t[k]
                arr = v.numpy() if hasattr(v, 'numpy') else np.array(v)
                break
        else:
            # pick first tensor-like value
            for v in t.values():
                if hasattr(v, 'numpy'):
                    arr = v.numpy()
                    break
            else:
                return None
    else:
        arr = np.array(t)

    # reduce dims
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        # fallback
        return None
    return arr


def gallery(preds_path: Path, test_csv: Path, spectrogram_base: Path, outdir: Path, mode: str = 'random', n: int = 25):
    ensure_outdir(outdir)
    preds = pd.read_csv(preds_path)
    test = pd.read_csv(test_csv)
    if len(preds) == len(test):
        preds = preds.copy()
        preds['filename'] = test['filename'].values
    else:
        print('Warning: preds and test lengths differ; filenames not attached')

    if mode == 'lowest':
        sel = preds.nsmallest(n, 'y_prob_max')
    else:
        sel = preds.sample(n, random_state=42)

    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.6))
    axes = axes.flatten()

    for ax in axes[n:]:
        ax.axis('off')

    for i, (_, row) in enumerate(sel.iterrows()):
        ax = axes[i]
        filename = row.get('filename')
        true_name = row['y_true_name']
        pred_name = row['y_pred_name']
        prob = row['y_prob_max']

        # spectrogram file expected at spectrogram_base/test/<true_name>/<filename>.pt
        spec_path = spectrogram_base / 'test' / true_name / f'{filename}.pt'
        arr = load_spectrogram_tensor(spec_path)
        if arr is None:
            ax.text(0.5, 0.5, 'no spectrogram', ha='center', va='center')
            ax.set_axis_off()
            continue

        # plot
        ax.imshow(arr, aspect='auto', origin='lower', cmap='magma')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(true_name, fontsize=8)
        caption = f'Pred: {pred_name} ({prob:.2f})'
        ax.text(0.5, -0.08, caption, transform=ax.transAxes, ha='center', fontsize=7)

    fig.suptitle(f'Gallery ({mode}) — {n} samples', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = outdir / f'gallery_{preds_path.stem}_{mode}_{n}.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print('Saved', out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, default='.', help='base path to model folder')
    parser.add_argument('--preds', type=str, default='outputs/baseline_preds_20251224_144210.csv')
    parser.add_argument('--test', type=str, default='../Bens-Internship-Local/Data/Annotations/test.csv')
    parser.add_argument('--history', type=str, default='baseline_best_training_history_1.csv')
    parser.add_argument('--report', type=str, default='outputs/baseline_classification_report_20251224_144210.json')
    parser.add_argument('--spectrograms', type=str, default='Data/Spectrograms')
    parser.add_argument('--out', type=str, default='outputs/analysis_plots')
    parser.add_argument('--gallery_mode', choices=['random', 'lowest'], default='random')
    parser.add_argument('--gallery_n', type=int, default=25)
    args = parser.parse_args()

    base = Path(args.base)
    outdir = base / args.out
    hist_path = base / args.history
    report_path = base / args.report
    preds_path = base / args.preds
    test_csv = Path(args.test)
    spectrogram_base = base / args.spectrograms

    # learning curves
    if hist_path.exists():
        plot_learning_curves([hist_path], ['baseline'], outdir)
    else:
        print('history file not found:', hist_path)

    # precision / recall
    if report_path.exists():
        plot_precision_recall_from_report(report_path, outdir)
    else:
        print('report file not found:', report_path)

    # strip plot
    if preds_path.exists() and test_csv.exists():
        plot_strip_predictions(preds_path, test_csv, outdir)
    else:
        print('preds or test csv not found')

    # gallery
    if preds_path.exists() and test_csv.exists():
        gallery(preds_path, test_csv, spectrogram_base, outdir, mode=args.gallery_mode, n=args.gallery_n)
    else:
        print('gallery skipped: preds/test missing')


if __name__ == '__main__':
    main()
