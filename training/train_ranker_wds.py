import os
import re
import math
import argparse
from typing import Dict, Any, List
import json
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import webdataset as wds

from tqdm import tqdm

from model.prm_scorer import PRM
from training.utils import freeze_bert_params


api_keys_json_path = os.path.join("./api_keys.json")
with open(api_keys_json_path) as f:
    data = json.load(f)
    os.environ["OPENAI_API_KEY"] = data["OPENAI_API_KEY"]
    if "HF_HOME" in data.keys():
        os.environ["HF_HOME"] = os.path.expanduser(data["HF_HOME"])
        
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

TRAIN_STEPS_PER_EPOCH = 200000 
VAL_STEPS_PER_EPOCH = 4090


def collate_pretokenized(batch: List[Dict[str, Any]], pad_token_id: int):
    """
    Handles dynamic padding for pre-tokenized sequences stored in shards.
    """
    # 1. Convert lists from JSON into Tensors
    pos_ids = [torch.tensor(x["input_ids_pos"]) for x in batch]
    pos_mask = [torch.tensor(x["attention_mask_pos"]) for x in batch]
    
    neg_ids = [torch.tensor(x["input_ids_neg"]) for x in batch]
    neg_mask = [torch.tensor(x["attention_mask_neg"]) for x in batch]
    
    w = torch.tensor([x["weight"] for x in batch], dtype=torch.float)

    # 2. Dynamic Padding (pads to the longest sequence in THIS batch only)
    # ModernBERT/BERT use 0 for attention mask padding and pad_token_id for input_ids
    enc_pos = {
        "input_ids": pad_sequence(pos_ids, batch_first=True, padding_value=pad_token_id),
        "attention_mask": pad_sequence(pos_mask, batch_first=True, padding_value=0),
    }
    
    enc_neg = {
        "input_ids": pad_sequence(neg_ids, batch_first=True, padding_value=pad_token_id),
        "attention_mask": pad_sequence(neg_mask, batch_first=True, padding_value=0),
    }

    return enc_pos, enc_neg, w


@torch.no_grad()
def eval_pairwise(model, loader, device, gap_thresholds: List[float]):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0

    thr_correct = {thr: 0 for thr in gap_thresholds}
    thr_total   = {thr: 0 for thr in gap_thresholds}


    for i, (enc_pos, enc_neg, w) in tqdm(enumerate(loader), total=VAL_STEPS_PER_EPOCH):
        enc_pos = {k: v.to(device) for k, v in enc_pos.items()}
        enc_neg = {k: v.to(device) for k, v in enc_neg.items()}
        w = w.to(device)

        s_pos = model(**enc_pos)
        s_neg = model(**enc_neg)

        diff = (s_pos - s_neg).view(-1)   # [B]
        gap  = w.view(-1)                # [B]

        beta = 5             # e.g., 5.0
        y = torch.sigmoid(beta * gap)    # in (0.5, 1.0)

        # BCE with logits; per-sample
        loss_vec = F.binary_cross_entropy_with_logits(diff, y, reduction="none")

        # loss_vec = F.softplus(-diff)

        # optional weighting: emphasize larger gaps but gently
        w_eff = torch.clamp(gap, min=0.0, max=1.0).pow(1.5)   # pow=0.5 or 1.0
        loss = (loss_vec * w_eff).mean()
        loss_sum += float(loss.item()) * len(w)

        # correct += int((diff > 0).sum().item())
        pred = (diff > 0)
        correct += int(pred.sum().item())
        total += len(w)

        # gap-bucket accuracies (using raw w/gap; not clamped)
        for thr in gap_thresholds:
            m = (w >= thr)
            if m.any():
                thr_correct[thr] += int((pred & m).sum().item())
                thr_total[thr]   += int(m.sum().item())

        if i == VAL_STEPS_PER_EPOCH:
            break 

    out = {
         "pair_acc": correct / max(1, total),
         "avg_loss": loss_sum / max(1, total),
     }
    
    for thr in gap_thresholds:
        out[f"pair_acc_gap_ge_{thr:g}"] = thr_correct[thr] / max(1, thr_total[thr])
        out[f"pair_n_gap_ge_{thr:g}"]   = thr_total[thr]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_prebuilt_pairs_path", type=str, default=None)
    ap.add_argument("--val_prebuilt_pairs_path", type=str, default=None)

    ap.add_argument("--cutoff_ratio", type=float, default=None)
    ap.add_argument("--weight_pct_leaf_correct", type=float, default=None)
    ap.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-base")
    ap.add_argument("--output_dir", type=str, default="./data/pairwise_prm")
    ap.add_argument("--freeze_bert", action="store_true")
    ap.add_argument("--freeze_first_encoder_num", type=int, default=18)

    ap.add_argument("--min_abs_gap", type=float, default=0.05)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--dropout_p", type=float, default=0.2)
    ap.add_argument("--truncation_side", type=str, default="left", choices=["left", "right"])
    ap.add_argument("--gap_acc_thresholds", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6")

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--warmup_ratio", type=float, default=0.08)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=5000)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--checkpoint_path", type=str, default="")

    args = ap.parse_args()

    # Extract cutoff_ratio and weight_pct_leaf_correct from the path if not provided
    if args.cutoff_ratio is None or args.weight_pct_leaf_correct is None:
        path = args.train_prebuilt_pairs_path
        m_cutoff = re.search(r'cutoff(\d+\.?\d*)', path)
        m_wc = re.search(r'_wc(\d+\.?\d*)', path)
        if m_cutoff is None or m_wc is None:
            raise ValueError(
                f"Cannot extract cutoff_ratio / weight_pct_leaf_correct from path: {path}. "
                "Expected format: ...cutoff<N>_wc<N>..."
            )
        if args.cutoff_ratio is None:
            args.cutoff_ratio = float(m_cutoff.group(1))
        if args.weight_pct_leaf_correct is None:
            args.weight_pct_leaf_correct = float(m_wc.group(1))

    gap_thresholds = []
    if args.gap_acc_thresholds.strip():
        gap_thresholds = sorted([float(x) for x in args.gap_acc_thresholds.split(",") if x.strip() != ""])

    args.w_correct = args.weight_pct_leaf_correct

    output_dir = os.path.join(args.output_dir, f"cf_{args.cutoff_ratio}_wcorrect_{args.w_correct}")
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model = PRM("./api_keys.json",
    #     args.model_name,
    #     dropout_p=args.dropout_p).to(device)

    model = PRM("./api_keys.json",
        args.model_name).to(device)

    if args.freeze_bert:
        model = freeze_bert_params(model, args.freeze_first_encoder_num)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    train_ds = (
        wds.WebDataset(args.train_prebuilt_pairs_path, resampled=True) # resampled=True for infinite training
        .shuffle(20000)
        .decode() # Automatically handles .json, .txt, etc.
        .map(lambda x: {
            "input_ids_pos": x["json"]["input_ids_pos"],
            "attention_mask_pos": x["json"]["attention_mask_pos"],
            "input_ids_neg": x["json"]["input_ids_neg"],
            "attention_mask_neg": x["json"]["attention_mask_neg"],
            "weight": x["json"]["weight"]
        })
        .with_epoch(TRAIN_STEPS_PER_EPOCH * args.batch_size)
    )

    val_ds = (
        wds.WebDataset(args.val_prebuilt_pairs_path, resampled=False) # resampled=True for infinite training
        .shuffle(20000)
        .decode() # Automatically handles .json, .txt, etc.
        .map(lambda x: {
            "input_ids_pos": x["json"]["input_ids_pos"],
            "attention_mask_pos": x["json"]["attention_mask_pos"],
            "input_ids_neg": x["json"]["input_ids_neg"],
            "attention_mask_neg": x["json"]["attention_mask_neg"],
            "weight": x["json"]["weight"]
        })
        .with_epoch(VAL_STEPS_PER_EPOCH * args.batch_size)
    )

    sample = next(iter(train_ds))
    print(sample)

    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=8,
        # Use the new collate function
        collate_fn=lambda b: collate_pretokenized(b, tokenizer.pad_token_id),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=8,
        collate_fn=lambda b: collate_pretokenized(b, tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = math.ceil(TRAIN_STEPS_PER_EPOCH / args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    ckpt = args.checkpoint_path

    if os.path.exists(ckpt):
        model_state = torch.load(ckpt)["model_state"]
        model.load_state_dict(model_state)
        print(f"Loaded model weights from {ckpt}")

        optimizer_state = torch.load(ckpt)["optimizer_state"]
        optimizer.load_state_dict(optimizer_state)
        print(f"Loaded optimizer state from {ckpt}")

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    best_acc = -1.0
    global_step = 0

            
    # running train stats since last log
    run_train_loss_sum = 0.0
    run_train_n = 0
    run_train_correct = 0

    run_train_thr_correct = {thr: 0 for thr in gap_thresholds}
    run_train_thr_total   = {thr: 0 for thr in gap_thresholds}

    all_histories = []
    history_path = os.path.join(output_dir, "train_history.jsonl")
    no_improve_count = 0

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for step, (enc_pos, enc_neg, w) in tqdm(enumerate(train_loader), total=TRAIN_STEPS_PER_EPOCH, desc=f"epoch {epoch}"):
            enc_pos = {k: v.to(device) for k, v in enc_pos.items()}
            enc_neg = {k: v.to(device) for k, v in enc_neg.items()}
            w = w.to(device)

            with torch.cuda.amp.autocast(enabled=args.fp16):
                s_pos = model(**enc_pos)
                s_neg = model(**enc_neg)

                diff = (s_pos - s_neg).view(-1)   # [B]
                gap  = w.view(-1)                # [B]

                beta = 5             # e.g., 5.0
                y = torch.sigmoid(beta * gap)    # in (0.5, 1.0)

                # BCE with logits; per-sample
                loss_vec = F.binary_cross_entropy_with_logits(diff, y, reduction="none")

                # loss_vec = F.softplus(-diff)

                # optional weighting: emphasize larger gaps but gently
                w_eff = torch.clamp(gap, min=0.0, max=1.0).pow(1.5)   # pow=0.5 or 1.0
                loss_for_logging = (loss_vec * w_eff).mean()
                loss = loss_for_logging / args.grad_accum

                pred = (diff > 0)
                batch_correct = pred.sum().item()
                batch_n = diff.numel()


            scaler.scale(loss).backward()

            # accumulate running train stats every micro-batch
            run_train_loss_sum += float(loss_for_logging.item()) * batch_n
            run_train_correct += int(batch_correct)
            run_train_n += int(batch_n)

            # gap-bucket running stats (use raw w/gap; not clamped)
            for thr in gap_thresholds:
                m = (w >= thr)
                if m.any():
                    run_train_thr_correct[thr] += int((pred & m).sum().item())
                    run_train_thr_total[thr]   += int(m.sum().item())

            # optimizer step
            if (step + 1) % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                # log/eval every 500 optimizer steps
                if global_step % args.log_every == 0:
                    train_loss = run_train_loss_sum / max(1, run_train_n)
                    train_acc = run_train_correct / max(1, run_train_n)

                    train_acc_by_gap = {
                        f"train_acc_gap_ge_{thr:g}": (run_train_thr_correct[thr] / max(1, run_train_thr_total[thr]))
                        for thr in gap_thresholds
                    }

                    # run val
                    # metrics = eval_pairwise(model, val_loader, device)
                    metrics = eval_pairwise(model, val_loader, device, gap_thresholds)
                    val_loss = metrics["avg_loss"]
                    val_acc = metrics["pair_acc"]
                    val_acc_by_gap = {k: v for k, v in metrics.items() if k.startswith("pair_acc_gap_ge_") or k.startswith("pair_n_gap_ge_")}

                    print(
                        f"[gs {global_step}] "
                        f"train acc={train_acc:.4f} loss={train_loss:.4f} | "
                        f"val acc={val_acc:.4f} loss={val_loss:.4f}"
                    )

                    for thr in gap_thresholds:
                        print(
                            f"    gap>={thr:g}: "
                            f"train_acc={train_acc_by_gap[f'train_acc_gap_ge_{thr:g}']:.4f} "
                            f"| val_acc={metrics[f'pair_acc_gap_ge_{thr:g}']:.4f} "
                            f"(val_n={metrics[f'pair_n_gap_ge_{thr:g}']})"
                        )

                    record = {
                         "epoch": epoch,
                         "global_step": global_step,
                         "train_loss": train_loss,
                         "train_acc": train_acc,
                         "val_loss": val_loss,
                         "val_acc": val_acc,
                    }
                    record.update(train_acc_by_gap)
                    record.update(val_acc_by_gap)
                    all_histories.append(record)

                    with open(history_path, "w") as f:
                        for record in all_histories:
                            f.write(json.dumps(record) + "\n")

                    # reset running stats window
                    run_train_loss_sum = 0.0
                    run_train_correct = 0
                    run_train_n = 0
                    run_train_thr_correct = {thr: 0 for thr in gap_thresholds}
                    run_train_thr_total   = {thr: 0 for thr in gap_thresholds}

                    # save best
                    if val_acc > best_acc:
                        best_acc = val_acc
                        save_path = os.path.join(output_dir, f"best_prm.pt")
                        torch.save({"model_state": model.state_dict(),
                                    "optimizer_state": optimizer.state_dict(),
                                    "scheduler_state": scheduler.state_dict(),
                                    "args": vars(args)}, save_path)
                        print(f"Saved best checkpoint to {save_path}")
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        if no_improve_count >= args.patience:
                            print("No improvement for several evals, early stopping.")
                            return

    print(f"Done. Best val pair_acc={best_acc:.4f}")
    print(f"Saved training curve to {history_path}")
    


if __name__ == "__main__":
    main()
