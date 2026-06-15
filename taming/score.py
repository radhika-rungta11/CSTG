"""Per-Gaussian importance score and densification-stat accumulation.

The score blends median-normalized Gaussian-level signals — view-space
gradient, opacity, scale and screen radii. (Taming's per-pixel rasterizer
accumulation terms are unavailable in CSTG's spacetime rasterizer; see the
package docstring.)
"""

import torch


def _normalize(value, multiplier):
    """Median-normalize a per-Gaussian signal (taming's normalize()).

    NaNs and non-positive entries map to 0; positive entries are scaled by
    ``multiplier * value / median(positive values)``.
    """
    v = value.detach().to(torch.float32).clone()
    v[v.isnan()] = 0.0
    out = torch.zeros_like(v)
    valid = v > 0
    if valid.any():
        med = torch.median(v[valid])
        if med > 0:
            out[valid] = multiplier * (v[valid] / med)
    return out


def compute_score(g, w_grad=1.0, w_opacity=1.0, w_scale=1.0, w_radii=1.0):
    """Per-Gaussian importance used to rank densification candidates."""
    grads = g.xyz_gradient_accum / g.denom
    grads[grads.isnan()] = 0.0
    grad_mag = torch.norm(grads, dim=-1)
    opacity = g.get_opacity.detach().squeeze(-1)
    scale = torch.prod(g.get_scaling.detach(), dim=1)
    radii = g.max_radii2D.detach()
    return (_normalize(grad_mag, w_grad)
            + _normalize(opacity, w_opacity)
            + _normalize(scale, w_scale)
            + _normalize(radii, w_radii))


@torch.no_grad()
def accumulate_densification_stats(g, viewspace_point_tensor, visibility_filter, radii):
    """Accumulate the view-space gradient + max screen radii the score needs.

    Call once per render inside the densification window (before the optimizer
    zeroes gradients). MCMC does not need this, so callers gate it on
    ``densify_mode == "taming"``.
    """
    vis = visibility_filter
    g.max_radii2D[vis] = torch.max(g.max_radii2D[vis], radii[vis])
    g.add_densification_stats(viewspace_point_tensor, vis)
