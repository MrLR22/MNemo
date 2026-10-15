# model/evaluate.py
# Evaluation and ablation study for EgoActionRecognizer.
#
# Produces:
#   1. Top-1 / Top-5 accuracy for verb and noun
#   2. Per-class precision, recall, F1
#   3. Confusion matrix (saved as image)
#   4. Ablation study: drop each modality, measure accuracy drop
#
# Usage:
#   python model/evaluate.py --features data/extracted_features.pt --checkpoint checkpoints/best_model.pt

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from collections import defaultdict

from model.architecture import EgoActionRecognizer
from model.train import EgoActionDataset


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_verbs: int,
    num_nouns: int,
) -> dict:
    """Compute full evaluation metrics."""
    model.eval()

    all_verb_preds = []
    all_verb_labels = []
    all_noun_preds = []
    all_noun_labels = []
    all_verb_logits = []
    all_noun_logits = []

    with torch.no_grad():
        for batch in dataloader:
            clip_f = batch["clip"].to(device)
            obj_f = batch["obj"].to(device)
            hand_f = batch["hand"].to(device)
            flow_f = batch["flow"].to(device)
            verb_y = batch["verb"]
            noun_y = batch["noun"]

            verb_logits, noun_logits = model(clip_f, obj_f, hand_f, flow_f)

            all_verb_preds.extend(verb_logits.argmax(1).cpu().tolist())
            all_noun_preds.extend(noun_logits.argmax(1).cpu().tolist())
            all_verb_labels.extend(verb_y.tolist())
            all_noun_labels.extend(noun_y.tolist())
            all_verb_logits.append(verb_logits.cpu())
            all_noun_logits.append(noun_logits.cpu())

    verb_preds = np.array(all_verb_preds)
    verb_labels = np.array(all_verb_labels)
    noun_preds = np.array(all_noun_preds)
    noun_labels = np.array(all_noun_labels)
    verb_logits_all = torch.cat(all_verb_logits)
    noun_logits_all = torch.cat(all_noun_logits)

    results = {
        "verb_top1": float((verb_preds == verb_labels).mean()),
        "noun_top1": float((noun_preds == noun_labels).mean()),
    }

    # Top-5
    _, verb_top5 = verb_logits_all.topk(min(5, num_verbs), dim=1)
    _, noun_top5 = noun_logits_all.topk(min(5, num_nouns), dim=1)
    verb_labels_t = torch.tensor(verb_labels).unsqueeze(1)
    noun_labels_t = torch.tensor(noun_labels).unsqueeze(1)
    results["verb_top5"] = float((verb_top5 == verb_labels_t).any(1).float().mean())
    results["noun_top5"] = float((noun_top5 == noun_labels_t).any(1).float().mean())

    # Per-class precision, recall, F1 for verbs
    results["verb_per_class"] = per_class_metrics(verb_labels, verb_preds, num_verbs)
    results["noun_per_class"] = per_class_metrics(noun_labels, noun_preds, num_nouns)

    # Confusion matrices
    results["verb_confusion"] = confusion_matrix(verb_labels, verb_preds, num_verbs)
    results["noun_confusion"] = confusion_matrix(noun_labels, noun_preds, num_nouns)

    return results


def per_class_metrics(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> list[dict]:
    """Compute precision, recall, F1 per class."""
    metrics = []
    for c in range(num_classes):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = int((labels == c).sum())
        metrics.append({
            "class": c, "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4), "support": support,
        })
    return metrics


def confusion_matrix(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> np.ndarray:
    """Standard confusion matrix: rows = true, cols = predicted."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(labels, preds):
        cm[true, pred] += 1
    return cm


# ── Ablation Study ─────────────────────────────────────────────────────────────

def run_ablation(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_verbs: int,
    num_nouns: int,
    frames_per_clip: int,
) -> list[dict]:
    """
    Ablation: zero out each modality one at a time and measure accuracy drop.
    Shows the contribution of each modality to the final result.
    """
    model.eval()
    modalities = ["clip", "obj", "hand", "flow"]
    dims = {"clip": 512, "obj": 80, "hand": 42, "flow": 8}

    results = []

    # Full model (all modalities)
    full_metrics = _ablation_evaluate(model, dataloader, device, None, num_verbs)
    results.append({
        "variant": "Full model (all modalities)",
        "verb_top1": full_metrics["verb_top1"],
        "noun_top1": full_metrics["noun_top1"],
        "verb_drop": 0.0,
    })

    # Drop each modality
    for mod in modalities:
        metrics = _ablation_evaluate(model, dataloader, device, mod, num_verbs)
        results.append({
            "variant": f"Without {mod.upper()}",
            "verb_top1": metrics["verb_top1"],
            "noun_top1": metrics["noun_top1"],
            "verb_drop": full_metrics["verb_top1"] - metrics["verb_top1"],
        })

    return results


def _ablation_evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    zero_modality: str | None,
    num_verbs: int,
) -> dict:
    """Evaluate with one modality zeroed out."""
    model.eval()
    verb_correct = 0
    noun_correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            clip_f = batch["clip"].to(device)
            obj_f = batch["obj"].to(device)
            hand_f = batch["hand"].to(device)
            flow_f = batch["flow"].to(device)

            # Zero out the specified modality
            if zero_modality == "clip":
                clip_f = torch.zeros_like(clip_f)
            elif zero_modality == "obj":
                obj_f = torch.zeros_like(obj_f)
            elif zero_modality == "hand":
                hand_f = torch.zeros_like(hand_f)
            elif zero_modality == "flow":
                flow_f = torch.zeros_like(flow_f)

            verb_logits, noun_logits = model(clip_f, obj_f, hand_f, flow_f)

            verb_correct += (verb_logits.argmax(1) == batch["verb"].to(device)).sum().item()
            noun_correct += (noun_logits.argmax(1) == batch["noun"].to(device)).sum().item()
            total += clip_f.size(0)

    return {
        "verb_top1": verb_correct / total,
        "noun_top1": noun_correct / total,
    }


# ── Confusion Matrix Visualization ─────────────────────────────────────────────

def save_confusion_matrix(cm: np.ndarray, class_names: list[str], title: str, path: str):
    """Save confusion matrix as an image using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.6), max(6, len(class_names) * 0.5)))

        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(title, fontsize=14)
        fig.colorbar(im, ax=ax, shrink=0.8)

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        # Annotate cells with counts
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j] > 0:
                    ax.text(j, i, str(cm[i, j]),
                            ha="center", va="center", fontsize=6,
                            color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")
    except ImportError:
        print("  [WARNING] matplotlib not installed — skipping confusion matrix image")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate EgoActionRecognizer")
    parser.add_argument("--features", type=str, required=True,
                        help="Path to extracted features .pt file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="evaluation_results")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation study (drop each modality)")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load dataset (use same val split as training)
    dataset = EgoActionDataset(args.features)
    num_verbs = dataset.config["top_n_verbs"]
    num_nouns = dataset.config["top_n_nouns"]

    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    _, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # same seed as training
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Load model
    checkpoint = torch.load(args.checkpoint, weights_only=False, map_location=device)
    embed_dim = checkpoint.get("embed_dim", 128)

    model = EgoActionRecognizer(
        num_verbs=num_verbs,
        num_nouns=num_nouns,
        embed_dim=embed_dim,
        max_clip_frames=dataset.config["frames_per_clip"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Build class name lookup from remap dicts
    verb_remap = checkpoint.get("verb_remap", dataset.verb_remap)
    noun_remap = checkpoint.get("noun_remap", dataset.noun_remap)
    verb_names = {v: k for k, v in verb_remap.items()} if verb_remap else {i: str(i) for i in range(num_verbs)}
    noun_names = {v: k for k, v in noun_remap.items()} if noun_remap else {i: str(i) for i in range(num_nouns)}

    # ── Full evaluation ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FULL EVALUATION")
    print(f"{'='*60}")

    metrics = compute_metrics(model, val_loader, device, num_verbs, num_nouns)

    print(f"\n  Verb Top-1 Accuracy:  {metrics['verb_top1']:.1%}")
    print(f"  Verb Top-5 Accuracy:  {metrics['verb_top5']:.1%}")
    print(f"  Noun Top-1 Accuracy:  {metrics['noun_top1']:.1%}")
    print(f"  Noun Top-5 Accuracy:  {metrics['noun_top5']:.1%}")

    # Per-class verb metrics
    print(f"\n  {'Verb Class':>12} | {'Prec':>6} | {'Recall':>6} | {'F1':>6} | {'Support':>7}")
    print(f"  {'-'*50}")
    for m in metrics["verb_per_class"]:
        name = str(verb_names.get(m["class"], m["class"]))
        print(f"  {name:>12} | {m['precision']:>6.1%} | {m['recall']:>6.1%} | "
              f"{m['f1']:>6.1%} | {m['support']:>7}")

    # Save confusion matrices
    os.makedirs(args.output_dir, exist_ok=True)
    verb_class_names = [str(verb_names.get(i, i)) for i in range(num_verbs)]
    noun_class_names = [str(noun_names.get(i, i)) for i in range(num_nouns)]

    save_confusion_matrix(
        metrics["verb_confusion"], verb_class_names,
        "Verb Confusion Matrix", os.path.join(args.output_dir, "verb_confusion.png")
    )
    save_confusion_matrix(
        metrics["noun_confusion"], noun_class_names,
        "Noun Confusion Matrix", os.path.join(args.output_dir, "noun_confusion.png")
    )

    # ── Ablation study ───────────────────────────────────────────────
    if args.ablation:
        print(f"\n{'='*60}")
        print("ABLATION STUDY")
        print(f"{'='*60}")

        ablation_results = run_ablation(
            model, val_loader, device, num_verbs, num_nouns,
            dataset.config["frames_per_clip"],
        )

        print(f"\n  {'Variant':<35} | {'Verb@1':>7} | {'Noun@1':>7} | {'Verb Drop':>10}")
        print(f"  {'-'*65}")
        for r in ablation_results:
            print(f"  {r['variant']:<35} | {r['verb_top1']:>6.1%} | "
                  f"{r['noun_top1']:>6.1%} | {r['verb_drop']:>+9.1%}")

        # Save ablation results
        ablation_path = os.path.join(args.output_dir, "ablation_results.txt")
        with open(ablation_path, "w") as f:
            f.write(f"{'Variant':<35} | {'Verb@1':>7} | {'Noun@1':>7} | {'Verb Drop':>10}\n")
            f.write(f"{'-'*65}\n")
            for r in ablation_results:
                f.write(f"{r['variant']:<35} | {r['verb_top1']:>6.1%} | "
                        f"{r['noun_top1']:>6.1%} | {r['verb_drop']:>+9.1%}\n")
        print(f"\n  Saved: {ablation_path}")

    print(f"\n[DONE] All results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
