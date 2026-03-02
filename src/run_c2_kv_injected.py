import json
import time
import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cases import CASES, MODEL_ID, resolve_model_source
from grader import build_results_payload, grade_answer
from train_c2_projector import SharedKVProjector, build_past_key_values, embed_skill_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="data/C2/001/projector.pt")
    parser.add_argument("--out", type=str, default="data/C2/001/results.json")
    parser.add_argument(
        "--alpha",
        type=float,
        default=-1.0,
        help="KV injection scale during eval. Negative uses checkpoint alpha_eval.",
    )
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    device = torch.device("cpu")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing projector checkpoint at {ckpt_path}. Run train_c2_projector.py first.")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    prefix_len = int(ckpt["prefix_len"])
    alpha_eval = float(ckpt.get("alpha_eval", 1.0)) if args.alpha < 0 else float(args.alpha)

    model_source = resolve_model_source(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_source, local_files_only=True).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    d_model = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    kv_heads = model.config.num_key_value_heads
    head_dim = d_model // num_heads

    projector = SharedKVProjector(
        in_dim=d_model,
        kv_heads=kv_heads,
        head_dim=head_dim,
        prefix_len=prefix_len,
    ).to(device)
    projector.load_state_dict(ckpt["projector_state_dict"])
    projector.eval()

    results = []
    total_score = 0.0
    total_time = 0.0

    for idx, case in enumerate(CASES, start=1):
        skill_text = Path(case["skill_file"]).read_text().strip()
        question = case["question"]
        expected = case["expected"]

        emb = embed_skill_text(model, tokenizer, skill_text, device)
        with torch.no_grad():
            projected = projector(emb) * alpha_eval
            pkv = build_past_key_values(projected, num_layers=num_layers, device=device)

            prompt = f"Q: {question}\nA:"
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            attn_len = input_ids.shape[1] + prefix_len
            attention_mask = torch.ones((1, attn_len), dtype=torch.long, device=device)

            t0 = time.perf_counter()
            out_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=pkv,
                max_new_tokens=24,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
            elapsed = time.perf_counter() - t0
            total_time += elapsed

        generated = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        answer = generated[len(prompt) :].strip().split("\n")[0].strip()
        grade = grade_answer(answer=answer, expected=expected, skill_text=skill_text)
        total_score += grade.final_score

        results.append(
            {
                "id": idx,
                "skill_file": case["skill_file"],
                "skill_key": grade.skill_key,
                "skill_value": grade.skill_value,
                "question": question,
                "expected": expected,
                "answer": answer,
                "semantic_match": grade.semantic_match,
                "quality_pass": grade.quality_pass,
                "degenerate_flag": grade.degenerate_flag,
                "correctness_component": grade.correctness_component,
                "non_degeneracy_component": grade.non_degeneracy_component,
                "final_score": grade.final_score,
                "scored_correct": grade.scored_correct,
                "latency_sec": round(elapsed, 4),
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_results_payload(results)
    out_path.write_text(json.dumps(payload, indent=2))

    avg_latency = total_time / len(CASES)
    print(f"C2 vector-KV score: {total_score:.1f}/{len(CASES)}")
    print(f"C2 avg latency: {avg_latency:.4f} sec/question")
    print(f"C2 alpha_eval: {alpha_eval:.4f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
