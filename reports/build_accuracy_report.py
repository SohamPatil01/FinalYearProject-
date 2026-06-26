"""Build accuracy/metrics artifacts for the report.

Reads the *real* validation metrics that Ultralytics embeds inside each trained
``.pt`` checkpoint (``train_metrics`` = best-epoch val scores, ``train_results`` =
per-epoch curves) and renders a table (CSV + Markdown) plus figures.

No labelled dataset is needed: these are the numbers each model recorded on its
own validation split during training. COCO-pretrained backbones (yolov8n / yolov10s)
have no custom val split here, so their official COCO benchmark mAP is shown for
reference and clearly labelled as such.

Run:  python reports/build_accuracy_report.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)

# (display name, file, role in the pipeline)
CUSTOM = [
    ("Truck detector", "truck.pt", "Truck restricted-hours"),
    ("Number-plate detector", "plate_best.pt", "Plate localization (for OCR)"),
    ("Helmet detector", "helmet_best.pt", "No-helmet rule"),
    ("Triple-seat detector", "triple.pt", "Triple-riding rule"),
]
# COCO-pretrained backbones used by the zone engines — official COCO val2017 mAP.
PRETRAINED = [
    ("YOLOv10-S", "yolov10s.pt", "Red-light engine", 0.463),
    ("YOLOv8-n", "yolov8n.pt", "No-parking + vehicle/rider scoping", 0.373),
]


def load_metrics(fname: str):
    ck = torch.load(MODELS / fname, map_location="cpu", weights_only=False)
    tm = ck.get("train_metrics", {}) or {}
    tr = ck.get("train_results", {}) or {}
    return tm, tr


def fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "—"


def main() -> None:
    rows = []
    curves = {}
    for name, fname, role in CUSTOM:
        tm, tr = load_metrics(fname)
        rows.append(
            {
                "Model": name,
                "Weights": fname,
                "Used for": role,
                "Precision": tm.get("metrics/precision(B)"),
                "Recall": tm.get("metrics/recall(B)"),
                "mAP@50": tm.get("metrics/mAP50(B)"),
                "mAP@50-95": tm.get("metrics/mAP50-95(B)"),
                "Source": "Trained val split",
            }
        )
        if tr.get("epoch") and tr.get("metrics/mAP50(B)"):
            curves[name] = (tr["epoch"], tr["metrics/mAP50(B)"])

    for name, fname, role, coco_map in PRETRAINED:
        rows.append(
            {
                "Model": name,
                "Weights": fname,
                "Used for": role,
                "Precision": None,
                "Recall": None,
                "mAP@50": None,
                "mAP@50-95": coco_map,
                "Source": "COCO benchmark (published)",
            }
        )

    # --- CSV + Markdown table ------------------------------------------------
    import csv

    cols = ["Model", "Weights", "Used for", "Precision", "Recall", "mAP@50", "mAP@50-95", "Source"]
    with open(OUT / "model_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (fmt(r[k]) if k in ("Precision", "Recall", "mAP@50", "mAP@50-95") else r[k]) for k in cols})

    md = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        md.append(
            "| "
            + " | ".join(
                fmt(r[c]) if c in ("Precision", "Recall", "mAP@50", "mAP@50-95") else str(r[c])
                for c in cols
            )
            + " |"
        )
    (OUT / "model_metrics.md").write_text("\n".join(md) + "\n")

    # --- Grouped bar chart (custom detectors) --------------------------------
    metrics = ["Precision", "Recall", "mAP@50", "mAP@50-95"]
    names = [r["Model"] for r in rows if r["Source"] == "Trained val split"]
    data = {m: [r[m] for r in rows if r["Source"] == "Trained val split"] for m in metrics}

    x = range(len(names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for i, m in enumerate(metrics):
        offs = [xi + (i - 1.5) * width for xi in x]
        bars = ax.bar(offs, data[m], width, label=m, color=colors[i])
        for b, v in zip(bars, data[m]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Per-model detection accuracy (validation metrics from training)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "detection_metrics_bar.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- Training curves (mAP@50 over epochs) --------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    for (name, (ep, mp)), c in zip(curves.items(), colors):
        ax.plot(ep, mp, label=name, color=c, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP@50")
    ax.set_title("Training convergence — mAP@50 vs epoch")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "training_curves_map50.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("Wrote:")
    for p in ("model_metrics.csv", "model_metrics.md", "detection_metrics_bar.png", "training_curves_map50.png"):
        print("  reports/" + p)
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
