import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cases import CASES, MODEL_ID, resolve_model_source
from grader import build_results_payload, grade_answer

def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    device = "cpu"
    model_source = resolve_model_source(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_source, local_files_only=True).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    total_score = 0.0
    total_time = 0.0

    for idx, case in enumerate(CASES, start=1):
        skill_text = Path(case["skill_file"]).read_text().strip()
        question = case["question"]
        expected = case["expected"]

        prompt = (
            "You are given one policy skill.\n"
            "Use only this skill to answer the question with the exact value.\n\n"
            f"Skill:\n{skill_text}\n\n"
            f"Q: {question}\nA:"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=24,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
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

    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "c1_results.json"
    payload = build_results_payload(results)
    out_path.write_text(json.dumps(payload, indent=2))

    avg_latency = total_time / len(CASES)
    print(f"C1 score: {total_score:.1f}/{len(CASES)}")
    print(f"C1 avg latency: {avg_latency:.4f} sec/question")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
