import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers.cache_utils import DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer

from cases import CASES, MODEL_ID, resolve_model_source


class SharedKVProjector(nn.Module):
    def __init__(self, in_dim: int, kv_heads: int, head_dim: int, prefix_len: int):
        super().__init__()
        out_dim = 2 * kv_heads * prefix_len * head_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.prefix_len = prefix_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [d_model]
        y = self.net(x)  # [2 * kv_heads * prefix_len * head_dim]
        y = y.view(2, self.kv_heads, self.prefix_len, self.head_dim)
        return y


def initialize_projector_near_zero(projector: SharedKVProjector, init_scale: float) -> None:
    # Keep output close to zero at startup so injected KV starts gentle.
    final_linear = projector.net[-1]
    nn.init.normal_(final_linear.weight, mean=0.0, std=init_scale)
    nn.init.zeros_(final_linear.bias)


def embed_skill_text(model, tokenizer, skill_text: str, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        ids = tokenizer(skill_text, return_tensors="pt").input_ids.to(device)
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        # last hidden state mean pooling: [1, seq, d] -> [d]
        emb = out.hidden_states[-1].mean(dim=1).squeeze(0)
        return emb.detach()


def build_past_key_values(
    projected: torch.Tensor,
    num_layers: int,
    device: torch.device,
) -> DynamicCache:
    # projected: [2, kv_heads, prefix_len, head_dim]
    k = projected[0].unsqueeze(0).to(device)  # [1, kv_heads, prefix_len, head_dim]
    v = projected[1].unsqueeze(0).to(device)
    legacy = tuple((k, v) for _ in range(num_layers))
    return DynamicCache.from_legacy_cache(legacy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--prefix-len", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--alpha-start", type=float, default=0.05)
    parser.add_argument("--alpha-end", type=float, default=0.25)
    parser.add_argument("--init-scale", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="data/C2/001/projector.pt")
    parser.add_argument("--history-out", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--resume-from", type=str, default="")
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    device = torch.device(args.device)
    model_source = resolve_model_source(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_source, local_files_only=True).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for p in model.parameters():
        p.requires_grad = False

    d_model = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    kv_heads = model.config.num_key_value_heads
    head_dim = d_model // num_heads

    projector = SharedKVProjector(
        in_dim=d_model,
        kv_heads=kv_heads,
        head_dim=head_dim,
        prefix_len=args.prefix_len,
    ).to(device)
    initialize_projector_near_zero(projector, init_scale=args.init_scale)

    optimizer = AdamW(projector.parameters(), lr=args.lr)

    start_epoch = 0
    history = []
    if args.resume_from:
        resume_path = Path(args.resume_from)
        ckpt = torch.load(resume_path, map_location="cpu")
        ckpt_prefix = int(ckpt.get("prefix_len", args.prefix_len))
        if ckpt_prefix != args.prefix_len:
            raise ValueError(
                f"resume checkpoint prefix_len={ckpt_prefix} does not match current --prefix-len={args.prefix_len}"
            )
        projector.load_state_dict(ckpt["projector_state_dict"])
        history = ckpt.get("history", [])
        start_epoch = int(history[-1]["epoch"]) if history else 0
        print(f"resumed from {resume_path} at epoch={start_epoch}")

    # Precompute fixed skill embeddings (no gradients through base model)
    skill_embeddings = {}
    for case in CASES:
        skill_text = Path(case["skill_file"]).read_text().strip()
        skill_embeddings[case["skill_file"]] = embed_skill_text(model, tokenizer, skill_text, device)

    num_cases = len(CASES)
    total_steps = max(1, args.epochs * num_cases)
    warmup_steps = max(0, min(args.warmup_steps, total_steps))
    global_step = 0

    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        total_loss = 0.0
        last_alpha = args.alpha_end
        last_lr = args.lr
        for case in CASES:
            question = case["question"]
            expected = case["expected"]
            emb = skill_embeddings[case["skill_file"]]

            global_step += 1
            if total_steps > 1:
                progress = float(global_step - 1) / float(total_steps - 1)
            else:
                progress = 1.0
            alpha = args.alpha_start + progress * (args.alpha_end - args.alpha_start)
            last_alpha = alpha

            if warmup_steps > 0 and global_step <= warmup_steps:
                lr_scale = float(global_step) / float(warmup_steps)
            else:
                lr_scale = 1.0
            current_lr = args.lr * lr_scale
            last_lr = current_lr
            for group in optimizer.param_groups:
                group["lr"] = current_lr

            projected = projector(emb) * alpha
            pkv = build_past_key_values(projected, num_layers=num_layers, device=device)

            prompt = f"Q: {question}\nA:"
            answer_text = f" {expected}"

            p_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            a_ids = tokenizer(answer_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            input_ids = torch.cat([p_ids, a_ids], dim=1)

            labels = torch.cat([torch.full_like(p_ids, -100), a_ids], dim=1)

            # Attention mask length includes cached prefix + current tokens.
            attn_len = input_ids.shape[1] + args.prefix_len
            attention_mask = torch.ones((1, attn_len), dtype=torch.long, device=device)

            outputs = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                past_key_values=pkv,
                use_cache=False,
            )
            loss = outputs.loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projector.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.detach().cpu())

        mean_loss = total_loss / len(CASES)
        history.append(
            {
                "epoch": epoch,
                "loss": mean_loss,
                "alpha": float(last_alpha),
                "lr": float(last_lr),
            }
        )
        print(f"epoch={epoch} loss={mean_loss:.4f} alpha={last_alpha:.4f} lr={last_lr:.6f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": MODEL_ID,
            "prefix_len": args.prefix_len,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "hidden_size": d_model,
            "lr": args.lr,
            "warmup_steps": args.warmup_steps,
            "grad_clip": args.grad_clip,
            "alpha_start": args.alpha_start,
            "alpha_end": args.alpha_end,
            "alpha_eval": args.alpha_end,
            "init_scale": args.init_scale,
            "projector_state_dict": projector.state_dict(),
            "history": history,
        },
        out_path,
    )

    hist_path = Path(args.history_out) if args.history_out else out_path.with_name("history.json")
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    hist_path.write_text(json.dumps(history, indent=2))
    print(f"saved projector: {out_path}")
    print(f"saved history: {hist_path}")


if __name__ == "__main__":
    main()
