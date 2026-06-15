"""Taming-3DGS steerable densification for CSTG.

Self-contained port of the budget-controlled, score-ranked densifier from
"Taming 3DGS: High-Quality Radiance Fields with Limited Resources"
(humansensinglab/taming-3dgs), adapted to CSTG's spacetime Gaussian model.

It is an *alternative* densifier, selected at runtime via the config key
``densify_mode`` ("mcmc" keeps the existing behaviour; "taming" uses this
package). Where MCMC reaches a target primitive count indirectly by tuning
``mcmc_cap_max`` + refine cadence, taming *sets* the final count directly via
``taming_budget`` — which is what shrinks the Optuna search space.

Why a subset of the upstream score
----------------------------------
Taming's full importance blends Gaussian-level signals (gradient, opacity,
scale, radii) with per-pixel rasterizer accumulation terms (blend/dist/loss/
count) emitted by its *modified* CUDA rasterizer. CSTG's spacetime rasterizer
does not expose those per-Gaussian accumulation outputs, so only the
Gaussian-level signals are used here. The deterministic budget schedule and the
score-ranked, budget-limited clone/split — the core "steerable" contribution —
are fully preserved.

Public API
----------
    get_count_array(start_count, budget, mode, opt)        -> list[int]
    compute_score(g, w_grad, w_opacity, w_scale, w_radii)  -> Tensor[N]
    densify_with_score(g, scores, budget, ...)             -> None (mutates g)
    densify_step(opt, g, iteration, scene)                 -> None  (driver)
    accumulate_densification_stats(g, vsp, vis, radii)     -> None

The functions take the gaussian model ``g`` as their first argument instead of
operating as methods, so this stays a drop-in package with no edits to the
GaussianModel class itself.
"""

from .budget import get_count_array
from .score import compute_score, accumulate_densification_stats
from .densify import densify_with_score, densify_and_clone, densify_and_split
from .step import densify_step

__all__ = [
    "get_count_array",
    "compute_score",
    "accumulate_densification_stats",
    "densify_with_score",
    "densify_and_clone",
    "densify_and_split",
    "densify_step",
]
