# SSS — Semantic Skill Space

Skill injection experiment for small models: compare markdown skill prompting vs skill-vector KV-cache injection.

## Hypothesis Statement

### Background

Skill files are a practical extension of prompt engineering: they inject human-language policy content into the model context window. This works, but it is token-expensive and less suitable for smaller models with limited context budgets.

This project tests whether skill semantics can be conveyed more efficiently by injecting vectors into KV cache space instead of injecting full markdown skill text.

### Research Question

For a fixed base model, does KV-cache skill injection preserve task-relevant skill behavior with lower context overhead than markdown skill prompting?

### Variables

- Independent variable (condition):
  - `C0`: no skill injection (base model only)
  - `C1`: markdown skill text injected into prompt context
  - `C2`: no markdown skill text in prompt; skill embedded and injected through projector -> KV cache
- Dependent variables:
  - correctness score
  - non-degeneracy score
  - combined final score

### Statistical Hypotheses

- Null hypothesis (`H0`):
  KV-cache injection does not provide meaningful skill-conditioned improvement over baseline (`C0`) and does not approach markdown skill performance (`C1`) on correctness and non-degeneracy.

- Alternative hypothesis (`H1`):
  KV-cache injection produces meaningful skill-conditioned improvement over baseline (`C0`) while maintaining non-degenerate responses, and narrows the performance gap with markdown skill prompting (`C1`) with lower prompt-context overhead.

### Primary Comparison Plan

Using the same frozen base model across all conditions:

1. Compare `C2` vs `C0` on correctness and overall quality.
2. Verify that `C2` remains usable (non-degenerate).
3. Compare `C2` vs `C1` as an efficiency-performance tradeoff baseline.

### Decision Guidance

Evidence supports `H1` if:

- `C2` is consistently above `C0` on correctness and final score,
- `C2` remains non-degenerate enough to be usable,
- and `C2` achieves this without markdown skill-text context injection.

Evidence supports `H0` if:

- `C2` remains near `C0`,
- or gains are mostly degenerate outputs,
- or `C2` is unstable/regressive across iterations.

## Model And Setup

- Base model: `Qwen/Qwen2.5-0.5B-Instruct` (frozen in all conditions)
- Dataset: 100 skill files, 1 question per skill
- Skill source: `skills/skill_*.md`
- Output files:
  - `data/c0_results.json`
  - `data/c1_results.json`
  - `data/C2/<iteration>/results.json`

## Grading

Per question:

- correctness component: `0.5`
- non-degeneracy component: `0.5`
- per-question total: `1.0`

Run-level summary fields:

- `correctness_out_of_50`
- `non_degeneracy_out_of_50`
- `final_score_out_of_100`

Degeneracy checks include repetition loops, numeric junk patterns, and garbled output checks.

## C2 Iteration Training

Each C2 iteration:

1. Read raw skill markdown text.
2. Embed skill text via frozen base model hidden states (mean pooled).
3. Project embedding into KV tensors.
4. Inject KV tensors during answer generation.
5. Train projector only (base model frozen).
6. Save checkpoint/history/results under `data/C2/<iteration>/`.
7. Next iteration resumes from previous checkpoint.

Current stabilization config:

- `epochs=3` per iteration
- `lr=1e-4`
- `prefix_len=2`
- warmup steps: `100`
- grad clip: `1.0`
- alpha ramp: `0.05 -> 0.25`
- near-zero projector init

## Results Tables

### Overall Comparison

| Condition | Correctness / 50 | Non-Degeneracy / 50 | Final / 100 |
|---|---:|---:|---:|
| C0 (no skills) | 4.0 | 46.0 | 50.0 |
| C1 (markdown skills) | 45.5 | 43.5 | 89.0 |
| C2-001 (first) | 1.5 | 19.5 | 21.0 |
| C2-005 (best) | 21.5 | 43.5 | 65.0 |
| C2-006 (latest) | 16.0 | 38.0 | 54.0 |

### C2 Iteration Progression

| C2 Iteration | Correctness / 50 | Non-Degeneracy / 50 | Final / 100 |
|---|---:|---:|---:|
| 001 | 1.5 | 19.5 | 21.0 |
| 002 | 10.0 | 29.0 | 39.0 |
| 003 | 18.5 | 40.0 | 58.5 |
| 004 | 21.0 | 40.0 | 61.0 |
| 005 | 21.5 | 43.5 | 65.0 |
| 006 | 16.0 | 38.0 | 54.0 |

### Knowledge (Correctness) Trend

| Step | Delta Correctness / 50 |
|---|---:|
| C0 -> 001 | -2.5 |
| 001 -> 002 | +8.5 |
| 002 -> 003 | +8.5 |
| 003 -> 004 | +2.5 |
| 004 -> 005 | +0.5 |
| 005 -> 006 | -5.5 |

### Non-Degeneracy Trend

| Step | Delta Non-Degeneracy / 50 |
|---|---:|
| C0 -> 001 | -26.5 |
| 001 -> 002 | +9.5 |
| 002 -> 003 | +11.0 |
| 003 -> 004 | +0.0 |
| 004 -> 005 | +3.5 |
| 005 -> 006 | -5.5 |

## Final Finding

This worked up to a clear point, then deteriorated:

- Benefit zone: `C2-001` through `C2-005`
  - final score improved from `21.0` to `65.0` (+44.0)
  - correctness improved from `1.5` to `21.5`
  - non-degeneracy improved from `19.5` to `43.5`
- Deterioration point: after `C2-005`
  - `C2-006` dropped to `54.0`
  - correctness and non-degeneracy both fell (`-5.5` each)

Interpretation:

- KV injection is viable for small models and can recover meaningful skill behavior when skill markdown context is not feasible in-window.
- It provides measurable gains up to a training point.
- Past that point, continued projector updates can overfit/diverge and reduce quality.

In practice, select the best checkpoint (here `005`) rather than always using the latest iteration.

## Run Commands

Controls:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /usr/bin/python3 src/run_c0_baseline.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /usr/bin/python3 src/run_c1_skill_scoped.py
```

C2:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /usr/bin/python3 src/run_c2_iterations.py --iteration 1 --python-bin /usr/bin/python3
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /usr/bin/python3 src/run_c2_iterations.py --iteration 2 --python-bin /usr/bin/python3
```
