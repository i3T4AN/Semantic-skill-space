import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def ensure_dirs() -> Path:
    data_root = Path("data/C2")
    data_root.mkdir(parents=True, exist_ok=True)
    return data_root


def iter_path(idx: int) -> Path:
    data_dir = Path("data/C2") / f"{idx:03d}"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def score_from_results(path: Path) -> tuple[int, int]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        total = int(data.get("total_questions", len(data.get("items", []))))
        points = float(
            data.get(
                "final_score_out_of_100",
                data.get("final_score_points", data.get("final_score_out_of_10", 0.0)),
            )
        )
        return points, total
    total = len(data)
    points = float(sum(float(row.get("final_score", 0.0)) for row in data))
    return points, total


def append_summary(
    idx: int,
    epochs: int,
    prefix_len: int,
    lr: float,
    warmup_steps: int,
    grad_clip: float,
    alpha_start: float,
    alpha_end: float,
    score: tuple[float, int],
    resume_from: str,
) -> None:
    summary_path = Path("data/C2/summary.jsonl")
    record = {
        "iteration": idx,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "epochs": epochs,
        "prefix_len": prefix_len,
        "lr": lr,
        "warmup_steps": warmup_steps,
        "grad_clip": grad_clip,
        "alpha_start": alpha_start,
        "alpha_end": alpha_end,
        "resume_from": resume_from,
        "score_points": score[0],
        "score_total": score[1],
    }
    with summary_path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def next_iteration_index() -> int:
    root = Path("data/C2")
    existing = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        try:
            existing.append(int(p.name))
        except Exception:
            continue
    return (max(existing) + 1) if existing else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--prefix-len", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--alpha-start", type=float, default=0.05)
    parser.add_argument("--alpha-end", type=float, default=0.25)
    parser.add_argument("--iteration", type=int, default=0, help="0 means auto-next")
    parser.add_argument("--python-bin", type=str, default=".venv/bin/python")
    args = parser.parse_args()

    ensure_dirs()

    idx = args.iteration if args.iteration > 0 else next_iteration_index()
    data_dir = iter_path(idx)

    iter_ckpt = data_dir / "projector.pt"
    iter_hist = data_dir / "history.json"
    iter_results = data_dir / "results.json"
    iter_meta = data_dir / "meta.json"
    prev_ckpt = (Path("data/C2") / f"{idx-1:03d}" / "projector.pt") if idx > 1 else None

    train_cmd = [
        args.python_bin,
        "src/train_c2_projector.py",
        "--epochs",
        str(args.epochs),
        "--prefix-len",
        str(args.prefix_len),
        "--lr",
        str(args.lr),
        "--warmup-steps",
        str(args.warmup_steps),
        "--grad-clip",
        str(args.grad_clip),
        "--alpha-start",
        str(args.alpha_start),
        "--alpha-end",
        str(args.alpha_end),
        "--out",
        str(iter_ckpt),
        "--history-out",
        str(iter_hist),
    ]
    resume_from = ""
    if prev_ckpt is not None and prev_ckpt.exists():
        train_cmd.extend(["--resume-from", str(prev_ckpt)])
        resume_from = str(prev_ckpt)
    run(train_cmd)

    run(
        [
            args.python_bin,
            "src/run_c2_kv_injected.py",
            "--checkpoint",
            str(iter_ckpt),
            "--out",
            str(iter_results),
        ]
    )

    score = score_from_results(iter_results)
    append_summary(
        idx=idx,
        epochs=args.epochs,
        prefix_len=args.prefix_len,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        alpha_start=args.alpha_start,
        alpha_end=args.alpha_end,
        score=score,
        resume_from=resume_from,
    )
    iter_meta.write_text(
        json.dumps(
            {
                "iteration": idx,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "epochs_added": args.epochs,
                "prefix_len": args.prefix_len,
                "lr": args.lr,
                "warmup_steps": args.warmup_steps,
                "grad_clip": args.grad_clip,
                "alpha_start": args.alpha_start,
                "alpha_end": args.alpha_end,
                "resume_from": resume_from,
                "score_points": score[0],
                "score_total": score[1],
            },
            indent=2,
        )
    )
    print(f"Iteration {idx} complete: C2 {score[0]}/{score[1]}")
    print(f"Saved checkpoint: {iter_ckpt}")
    print(f"Saved history: {iter_hist}")
    print(f"Saved results: {iter_results}")


if __name__ == "__main__":
    main()
