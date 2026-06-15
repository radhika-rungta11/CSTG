# taming/ — Taming-3DGS steerable densification

Self-contained port of the budget-controlled, score-ranked densifier from
[Taming 3DGS](https://github.com/humansensinglab/taming-3dgs), adapted to
CSTG's spacetime Gaussian model. It is an **alternative** to the default MCMC
densifier, selected at runtime — nothing changes unless you opt in.

## Why
MCMC reaches a target primitive count *indirectly* (you tune `mcmc_cap_max` +
refine cadence and see what count you land on). Taming **sets the final count
directly** via a deterministic per-step budget schedule. That turns the whole
"how many Gaussians" question from an Optuna search dimension into a single
knob, so the hyperparameter search gets smaller and the count becomes
predictable.

## Layout
| file | what |
|------|------|
| `budget.py` | `get_count_array()` — deterministic quadratic count ramp (taming Eq. 2) |
| `score.py`  | `compute_score()` per-Gaussian importance; `accumulate_densification_stats()` |
| `densify.py`| `densify_with_score()` + budget-limited `densify_and_clone/split` |
| `step.py`   | `densify_step()` — per-iteration driver (lays out + follows the schedule) |
| `__init__.py` | public API re-exports |

All functions take the gaussian model `g` as their first argument, so the
GaussianModel class is left untouched.

## How it's wired in
- `helper_train.py` → `_densify_refine_step()` dispatches to `taming.densify_step`
  when `densify_mode == "taming"`, else the existing MCMC step.
- `train.py` calls `taming.accumulate_densification_stats(...)` per render inside
  the densification window (taming needs view-space grad + screen radii; MCMC
  does not, so it's gated).
- `script/optuna_tuner.py` picks `densify_mode` and, for taming, sweeps
  `taming_budget` instead of `mcmc_cap_max`.

## Config keys (OptimizationParams)
```jsonc
{
  "densify_mode": "taming",          // "mcmc" (default) | "taming"
  "taming_budget": 1500000,          // multiplier of SfM count, or absolute count
  "taming_budget_mode": "final_count", // "multiplier" | "final_count"
  "taming_grad_threshold": 0.0002,
  "taming_min_opacity": 0.005,
  "taming_prune": 1,                 // 1 = score-weighted opacity/size prune
  "taming_w_grad": 1.0,              // score weights (Gaussian-level signals)
  "taming_w_opacity": 1.0,
  "taming_w_scale": 1.0,
  "taming_w_radii": 1.0
}
```
The densification window comes from the existing `densify_from_iter` /
`densify_until_iter` / `densification_interval`.

## Limitation
Taming's full score also blends per-pixel rasterizer accumulation terms
(blend/dist/loss/count) from its modified CUDA rasterizer. CSTG's spacetime
rasterizer does not expose those, so only the Gaussian-level signals (gradient,
opacity, scale, radii) are used. The deterministic budget and score-ranked
clone/split — the core steerable contribution — are preserved.
