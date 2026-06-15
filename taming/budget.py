"""Deterministic per-densification-step Gaussian-count schedule.

Implements the steerable budget ramp from taming-3dgs (Eq. 2): a quadratic
interpolation from the initial primitive count up to the requested budget,
sampled once per densification event over the densification window.
"""


def get_count_array(start_count, budget, mode, opt):
    """Return the list of per-step target counts.

    Args:
        start_count: live primitive count when densification begins.
        budget: multiplier of ``start_count`` (mode="multiplier") or the
            absolute final count (mode="final_count").
        mode: "multiplier" | "final_count".
        opt: optimization params; uses ``densify_from_iter``,
            ``densify_until_iter`` and ``densification_interval`` to size the
            number of densification events.

    The schedule is monotonic, starts at ``start_count`` and ends at the
    resolved target. Length is ``num_steps + 1`` so the driver can index
    ``counts[step + 1]`` safely.
    """
    if mode == "multiplier":
        target = int(start_count * float(budget))
    elif mode == "final_count":
        target = int(budget)
    else:
        raise ValueError(f"unknown taming_budget_mode '{mode}'")

    num_steps = max(1, (opt.densify_until_iter - opt.densify_from_iter) // opt.densification_interval)
    slope_lower_bound = (target - start_count) / num_steps

    k = 2 * slope_lower_bound
    a = (target - start_count - k * num_steps) / (num_steps * num_steps)
    b = k
    c = start_count

    return [int(a * (x ** 2) + b * x + c) for x in range(num_steps + 1)]
