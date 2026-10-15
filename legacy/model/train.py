# model/train.py
# Training script for EgoActionRecognizer.
# Reads pre-extracted features (.pt file from feature_extractor.py),
# trains the multi-modal temporal model, saves checkpoints.
#
# Usage:
#   python model/train.py --features data/extracted_features.pt --epochs 50
#   python model/train.py --features data/extracted_features.pt --device cuda  # on Colab

import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from model.architecture import EgoActionRecognizer


# ── Dataset ────────────────────────────────────────────────────────────────────

class EgoActionDataset(Dataset):
    """Loads pre-extracted multi-modal features for training."""

    def __init__(self, features_path: str):
        data = torch.load(features_path, weights_only=False)
        self.clip_features = data["clip_features"]    # (N, T, 512)
        self.obj_features = data["obj_features"]      # (N, T, 80)
        self.hand_features = data["hand_features"]    # (N, T, 42)
        self.flow_features = data["flow_features"]    # (N, T, 8)
        self.verb_labels = data["verb_labels"]        # (N,)
        self.noun_labels = data["noun_labels"]        # (N,)
        self.verb_remap = data["verb_remap"]
        self.noun_remap = data["noun_remap"]
        self.config = data["config"]

        print(f"[Dataset] Loaded {len(self)} samples")
        print(f"  Frames per clip: {self.config['frames_per_clip']}")
        print(f"  Verb classes: {self.config['top_n_verbs']}")
        print(f"  Noun classes: {self.config['top_n_nouns']}")

    def __len__(self):
        return len(self.verb_labels)

    def __getitem__(self, idx):
        return {
            "clip": self.clip_features[idx],
            "obj": self.obj_features[idx],
            "hand": self.hand_features[idx],
            "flow": self.flow_features[idx],
            "verb": self.verb_labels[idx],
            "noun": self.noun_labels[idx],
        }


# ── Training Loop ──────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    verb_criterion: nn.Module,
    noun_criterion: nn.Module,
    device: torch.device,
    verb_loss_weight: float = 0.6,
    noun_loss_weight: float = 0.4,
) -> dict:
    """Train for one epoch. Returns loss and accuracy metrics."""
    model.train()
    total_loss = 0.0
    verb_correct = 0
    noun_correct = 0
    total = 0

    for batch in dataloader:
        clip_f = batch["clip"].to(device)
        obj_f = batch["obj"].to(device)
        hand_f = batch["hand"].to(device)
        flow_f = batch["flow"].to(device)
        verb_y = batch["verb"].to(device)
        noun_y = batch["noun"].to(device)

        optimizer.zero_grad()
        verb_logits, noun_logits = model(clip_f, obj_f, hand_f, flow_f)

        loss_v = verb_criterion(verb_logits, verb_y)
        loss_n = noun_criterion(noun_logits, noun_y)
        loss = verb_loss_weight * loss_v + noun_loss_weight * loss_n

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * clip_f.size(0)
        verb_correct += (verb_logits.argmax(1) == verb_y).sum().item()
        noun_correct += (noun_logits.argmax(1) == noun_y).sum().item()
        total += clip_f.size(0)

    return {
        "loss": total_loss / total,
        "verb_acc": verb_correct / total,
        "noun_acc": noun_correct / total,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    verb_criterion: nn.Module,
    noun_criterion: nn.Module,
    device: torch.device,
    verb_loss_weight: float = 0.6,
    noun_loss_weight: float = 0.4,
) -> dict:
    """Evaluate on validation set. Returns loss and accuracy metrics."""
    model.eval()
    total_loss = 0.0
    verb_correct = 0
    noun_correct = 0
    verb_top5_correct = 0
    noun_top5_correct = 0
    total = 0

    for batch in dataloader:
        clip_f = batch["clip"].to(device)
        obj_f = batch["obj"].to(device)
        hand_f = batch["hand"].to(device)
        flow_f = batch["flow"].to(device)
        verb_y = batch["verb"].to(device)
        noun_y = batch["noun"].to(device)

        verb_logits, noun_logits = model(clip_f, obj_f, hand_f, flow_f)

        loss_v = verb_criterion(verb_logits, verb_y)
        loss_n = noun_criterion(noun_logits, noun_y)
        loss = verb_loss_weight * loss_v + noun_loss_weight * loss_n

        total_loss += loss.item() * clip_f.size(0)
        verb_correct += (verb_logits.argmax(1) == verb_y).sum().item()
        noun_correct += (noun_logits.argmax(1) == noun_y).sum().item()

        # Top-5 accuracy
        _, verb_top5 = verb_logits.topk(5, dim=1)
        _, noun_top5 = noun_logits.topk(5, dim=1)
        verb_top5_correct += (verb_top5 == verb_y.unsqueeze(1)).any(1).sum().item()
        noun_top5_correct += (noun_top5 == noun_y.unsqueeze(1)).any(1).sum().item()

        total += clip_f.size(0)

    return {
        "loss": total_loss / total,
        "verb_acc": verb_correct / total,
        "noun_acc": noun_correct / total,
        "verb_top5_acc": verb_top5_correct / total,
        "noun_top5_acc": noun_top5_correct / total,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train EgoActionRecognizer")
    parser.add_argument("--features", type=str, required=True,
                        help="Path to extracted features .pt file")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto",
                        help="'cpu', 'cuda', or 'auto'")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (epochs)")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[Train] Using device: {device}")

    # Load dataset
    dataset = EgoActionDataset(args.features)
    num_verbs = dataset.config["top_n_verbs"]
    num_nouns = dataset.config["top_n_nouns"]
    frames_per_clip = dataset.config["frames_per_clip"]

    # Train/val split
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"[Train] Train: {train_size}, Val: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = EgoActionRecognizer(
        num_verbs=num_verbs,
        num_nouns=num_nouns,
        embed_dim=args.embed_dim,
        max_clip_frames=frames_per_clip,
    ).to(device)
    print(f"[Train] Model parameters: {model.count_parameters():,}")

    # Loss with class weighting could be added here if classes are imbalanced
    verb_criterion = nn.CrossEntropyLoss()
    noun_criterion = nn.CrossEntropyLoss()

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_verb_acc = 0.0
    patience_counter = 0

    print(f"\n{'='*80}")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'T-Verb%':>8} | {'T-Noun%':>8} | "
          f"{'Val Loss':>10} | {'V-Verb%':>8} | {'V-Noun%':>8} | {'V-V@5%':>7} | {'LR':>10}")
    print(f"{'='*80}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, verb_criterion, noun_criterion, device
        )
        val_metrics = evaluate(
            model, val_loader, verb_criterion, noun_criterion, device
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(f"{epoch:>5} | {train_metrics['loss']:>10.4f} | {train_metrics['verb_acc']:>7.1%} | "
              f"{train_metrics['noun_acc']:>7.1%} | {val_metrics['loss']:>10.4f} | "
              f"{val_metrics['verb_acc']:>7.1%} | {val_metrics['noun_acc']:>7.1%} | "
              f"{val_metrics['verb_top5_acc']:>6.1%} | {lr:>10.6f}  ({elapsed:.1f}s)")

        # Save best model
        if val_metrics["verb_acc"] > best_val_verb_acc:
            best_val_verb_acc = val_metrics["verb_acc"]
            patience_counter = 0
            save_path = os.path.join(args.save_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "verb_remap": dataset.verb_remap,
                "noun_remap": dataset.noun_remap,
                "config": dataset.config,
                "embed_dim": args.embed_dim,
            }, save_path)
            print(f"  → Saved best model (verb_acc={best_val_verb_acc:.1%})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n[Train] Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    # Save final model too
    final_path = os.path.join(args.save_dir, "final_model.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_metrics": val_metrics,
        "verb_remap": dataset.verb_remap,
        "noun_remap": dataset.noun_remap,
        "config": dataset.config,
        "embed_dim": args.embed_dim,
    }, final_path)

    print(f"\n{'='*80}")
    print(f"Training complete.")
    print(f"  Best val verb accuracy: {best_val_verb_acc:.1%}")
    print(f"  Best model: {os.path.join(args.save_dir, 'best_model.pt')}")
    print(f"  Final model: {final_path}")


if __name__ == "__main__":
    main()
