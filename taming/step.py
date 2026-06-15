"""Per-iteration driver for the taming densifier.

Lays out the budget schedule once (lazily, from the live primitive count) and
then runs one score-ranked densification toward the next target whenever the
iteration falls on the densification cadence inside the densification window.
State (``_taming_counts``, ``_taming_step``) is stashed on the gaussian model
so ``controlgaussians`` can stay stateless.
"""

from .budget import get_count_array
from .score import compute_score
from .densify import densify_with_score


def densify_step(opt, g, iteration, scene):
    if (iteration <= opt.densify_from_iter
            or iteration > opt.densify_until_iter
            or iteration >= opt.iterations
            or iteration % opt.densification_interval != 0):
        return

    if getattr(g, "_taming_counts", None) is None:
        start = g.get_xyz.shape[0]
        g._taming_counts = get_count_array(
            start, opt.taming_budget, opt.taming_budget_mode, opt)
        g._taming_step = 0
        scene.recordpoints(
            iteration,
            f"taming schedule start={start} target={g._taming_counts[-1]} "
            f"steps={len(g._taming_counts)}")

    idx = min(g._taming_step + 1, len(g._taming_counts) - 1)
    target = g._taming_counts[idx]

    scores = compute_score(
        g, opt.taming_w_grad, opt.taming_w_opacity,
        opt.taming_w_scale, opt.taming_w_radii)
    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

    densify_with_score(
        g, scores=scores, budget=target, min_opacity=opt.taming_min_opacity,
        extent=scene.cameras_extent, max_screen_size=size_threshold,
        grad_threshold=opt.taming_grad_threshold, do_prune=bool(opt.taming_prune),
        iter_num=g._taming_step)
    g._taming_step += 1

    if iteration % (opt.densification_interval * 10) == 0:
        scene.recordpoints(
            iteration,
            f"taming densify step={g._taming_step} target={target} "
            f"total={g.get_xyz.shape[0]}")
