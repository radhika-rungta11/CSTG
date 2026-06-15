"""Score-ranked, budget-limited clone/split for CSTG's spacetime model.

These operate on the gaussian model ``g`` (passed in, not ``self``) and reuse
its ``densification_postfix`` / ``prune_points`` so all spacetime attributes
(trbf center/scale, motion, omega, temporal feature, mask) stay consistent.
"""

import torch
from utils.general_utils import build_rotation


def _sample(g, weights, qualifier_mask, budget):
    """Sample up to ``budget`` indices without replacement, weighted by
    ``weights``, restricted to entries where ``qualifier_mask`` is True and the
    weight is strictly positive.

    ``weights`` / ``qualifier_mask`` may be shorter than the live count (clone
    runs before split and appends points): any index beyond their length is a
    freshly-added point and is given zero weight so it is never sampled.
    """
    n = g.get_xyz.shape[0]
    sel = torch.zeros(n, dtype=torch.bool, device="cuda")
    w = torch.zeros(n, dtype=torch.float32, device="cuda")
    qm = torch.zeros(n, dtype=torch.bool, device="cuda")
    mw = min(n, weights.shape[0])
    w[:mw] = weights.detach().to(torch.float32).reshape(-1)[:mw]
    mq = min(n, qualifier_mask.shape[0])
    qm[:mq] = qualifier_mask.reshape(-1)[:mq]
    w[~qm] = 0.0
    w[w.isnan()] = 0.0
    w = torch.clamp(w, min=0.0)
    nonzero = int((w > 0).sum().item())
    k = int(min(int(budget), nonzero))
    if k <= 0:
        return sel
    idx = torch.multinomial(w, k, replacement=False)
    sel[idx] = True
    return sel


def densify_and_clone(g, scores, budget, qualifier_mask):
    """Duplicate up to ``budget`` clone-qualifiers, picked by importance."""
    selected = _sample(g, scores, qualifier_mask, budget)
    if selected.sum() == 0:
        return
    new_xyz = g._xyz[selected]
    new_opacities = g._opacity[selected]
    new_scaling = g._scaling[selected]
    new_rotation = g._rotation[selected]
    new_trbf_center = torch.rand((int(selected.sum()), 1), device="cuda")
    new_trbfscale = g._trbf_scale[selected]
    new_motion = g._motion[selected]
    new_omega = g._omega[selected]
    new_featuret = g._features_t[selected]
    new_mask = g._mask[selected]
    g.densification_postfix(new_xyz, new_opacities, new_scaling, new_rotation,
                            new_trbf_center, new_trbfscale, new_motion,
                            new_omega, new_featuret, new_mask)


def densify_and_split(g, scores, budget, qualifier_mask, N=2):
    """Split up to ``budget`` split-qualifiers into N children, by importance."""
    selected = _sample(g, scores, qualifier_mask, budget)
    n_sel = int(selected.sum())
    if n_sel == 0:
        return
    stds = g.get_scaling[selected].repeat(N, 1)
    means = torch.zeros((stds.size(0), 3), device="cuda")
    samples = torch.normal(mean=means, std=stds)
    rots = build_rotation(g._rotation[selected]).repeat(N, 1, 1)
    new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + g.get_xyz[selected].repeat(N, 1)
    new_scaling = g.scaling_inverse_activation(g.get_scaling[selected].repeat(N, 1) / (0.8 * N))
    new_rotation = g._rotation[selected].repeat(N, 1)
    new_opacity = g._opacity[selected].repeat(N, 1)
    new_trbf_center = torch.rand_like(g._trbf_center[selected].repeat(N, 1))
    new_trbf_scale = g._trbf_scale[selected].repeat(N, 1)
    new_motion = g._motion[selected].repeat(N, 1)
    new_omega = g._omega[selected].repeat(N, 1)
    new_features_t = g._features_t[selected].repeat(N, 1)
    new_mask = g._mask[selected].repeat(N, 1)
    g.densification_postfix(new_xyz, new_opacity, new_scaling, new_rotation,
                            new_trbf_center, new_trbf_scale, new_motion,
                            new_omega, new_features_t, new_mask)
    prune_filter = torch.cat((selected, torch.zeros(N * n_sel, device="cuda", dtype=bool)))
    g.prune_points(prune_filter)


def densify_with_score(g, scores, budget, min_opacity, extent, max_screen_size,
                       grad_threshold=0.0002, do_prune=True, iter_num=0):
    """Grow toward ``budget`` total Gaussians.

    Headroom (budget - current) is split between clone- and split-qualifiers in
    proportion to their counts, and within each group candidates are chosen by
    importance ``scores``. Optionally thins half of the low-opacity / oversized
    Gaussians, weighting removal toward low-score points (high-score points and
    freshly-added points are protected).
    """
    grad_vars = g.xyz_gradient_accum / g.denom
    grad_vars[grad_vars.isnan()] = 0.0
    grad_qual = torch.norm(grad_vars, dim=-1) >= grad_threshold
    max_scale = torch.max(g.get_scaling, dim=1).values
    clone_qual = torch.logical_and(grad_qual, max_scale <= g.percent_dense * extent)
    split_qual = torch.logical_and(grad_qual, max_scale > g.percent_dense * extent)
    total_clones = int(clone_qual.sum())
    total_splits = int(split_qual.sum())

    curr = g.get_xyz.shape[0]
    budget = min(int(budget), total_clones + total_splits + curr)
    extra = max(0, budget - curr)
    denom = max(1, total_clones + total_splits)
    clone_budget = (extra * total_clones) // denom
    split_budget = (extra * total_splits) // denom

    if clone_budget > 0:
        densify_and_clone(g, scores.clone(), clone_budget, clone_qual)
    if split_budget > 0:
        densify_and_split(g, scores.clone(), split_budget, split_qual)

    if do_prune:
        prune_mask = (g.get_opacity < min_opacity).squeeze(-1)
        if max_screen_size:
            big_points_vs = g.max_radii2D > max_screen_size
            big_points_ws = g.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        to_remove = int(prune_mask.sum())
        remove_budget = int(0.5 * to_remove)   # only thin out half, keep the rest
        if remove_budget > 0:
            n = g.get_xyz.shape[0]
            inv_importance = torch.zeros(n, dtype=torch.float32, device="cuda")
            m = min(n, scores.shape[0])
            # low score -> high removal weight; freshly added points keep 0 (protected)
            inv_importance[:m] = 1.0 / (1e-6 + scores[:m].reshape(-1).to(torch.float32))
            inv_importance[~prune_mask] = 0.0
            nonzero = int((inv_importance > 0).sum())
            k = int(min(remove_budget, nonzero))
            if k > 0:
                sampled = torch.multinomial(inv_importance, k, replacement=False)
                final_prune = torch.zeros(n, dtype=torch.bool, device="cuda")
                final_prune[sampled] = True
                g.prune_points(torch.logical_and(prune_mask, final_prune))

    torch.cuda.empty_cache()
