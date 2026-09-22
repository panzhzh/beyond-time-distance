"""Gaussian-copula quadrature with identical nine-knot marginal reconstructions."""
import torch


def inverse_quantiles(knots, probability):
    """Linear between deciles, logarithmic tails beyond 0.1 and 0.9.

    ``knots`` is B x H x 9, sorted. Probabilities broadcast to B x H x M
    and must lie strictly between zero and one. This reconstructs a CDF;
    the same reconstruction is shared by every dependence head.
    """
    u = probability.expand(knots.shape[0], knots.shape[1], -1)
    position = 10 * u - 1
    index = position.floor().long().clamp(0, 7)
    expanded = knots[:, :, None, :].expand(-1, -1, u.shape[-1], -1)
    low = torch.gather(expanded, 3, index[:, :, :, None]).squeeze(-1)
    high = torch.gather(expanded, 3, (index + 1)[:, :, :, None]).squeeze(-1)
    value = low + (position - index) * (high - low)
    lower = knots[:, :, 0, None] + (knots[:, :, 1, None] - knots[:, :, 0, None]) * (u / .1).log()
    upper = knots[:, :, 8, None] - (knots[:, :, 8, None] - knots[:, :, 7, None]) * ((1 - u) / .1).log()
    return torch.where(u < .1, lower, torch.where(u > .9, upper, value))


def empirical_crps(samples, target):
    count = samples.shape[-1]
    ordered = samples.sort(-1).values
    coefficient = 2 * torch.arange(count, device=samples.device, dtype=samples.dtype) + 1 - count
    return (samples - target[:, :, None]).abs().mean(-1) - (ordered * coefficient).sum(-1) / (count * count)


def balanced_uniforms(count, seed, device):
    sobol = torch.quasirandom.SobolEngine(2, scramble=True, seed=seed).draw(count, dtype=torch.float64).to(device)
    z = torch.special.ndtri(sobol.clamp(1e-12, 1 - 1e-12))
    rank = z[:, 0].argsort().argsort()
    return z[:, 0], z[:, 1], (rank.to(torch.float64) + .5) / count


def correlated_uniforms(correlation, z0, z1):
    rho = correlation.clamp(-1, 1)
    z = rho[:, :, None] * z0 + (1 - rho.square()).clamp_min(0).sqrt()[:, :, None] * z1
    order = z.argsort(-1)
    ranks = torch.empty_like(order)
    ranks.scatter_(-1, order, torch.arange(len(z0), device=z.device)[None, None, :].expand_as(order))
    return (ranks.to(torch.float64) + .5) / len(z0)


@torch.no_grad()
def reconstructed_evidence(knots, correlations, target, scale, pairs, quadrature, diagnostics=()):
    """Score reconstructed changes and retain absolute 90% interval diagnostics."""
    count = quadrature["count"]
    if count < 2 or count & (count - 1) or quadrature["batch_size"] < 1:
        raise ValueError("Use a positive batch size and power-of-two quadrature size")
    z0, z1, u0 = balanced_uniforms(count, quadrature["seed"], knots.device)
    scores = {name: knots.new_empty((len(knots), len(pairs.left))) for name in correlations}
    output = {f"{name}__{metric}": knots.new_empty((len(knots), len(pairs.left)))
              for name in diagnostics for metric in ("coverage90", "width90", "interval_score90")}
    probability = (0.05, 0.95)
    for start in range(0, len(knots), quadrature["batch_size"]):
        stop = min(start + quadrature["batch_size"], len(knots))
        q = knots[start:stop]
        left = inverse_quantiles(q, u0[None, None])[:, pairs.left]
        actual = target[start:stop, pairs.right] - target[start:stop, pairs.left]
        for name, correlation in correlations.items():
            u = correlated_uniforms(correlation[start:stop], z0, z1)
            changes = inverse_quantiles(q[:, pairs.right], u) - left
            scores[name][start:stop] = empirical_crps(changes, actual) / scale[start:stop, None]
            if name not in diagnostics:
                continue
            ordered = changes.sort(-1).values
            bounds = []
            for p in probability:
                position = p * (count - 1)
                low = int(position)
                bounds.append(ordered[:, :, low] * (1 - (position - low))
                              + ordered[:, :, low + 1] * (position - low))
            lo, hi = bounds
            width = hi - lo
            interval = width + 20 * (lo - actual).clamp_min(0) + 20 * (actual - hi).clamp_min(0)
            output[f"{name}__coverage90"][start:stop] = ((actual >= lo) & (actual <= hi)).to(knots.dtype)
            output[f"{name}__width90"][start:stop] = width
            output[f"{name}__interval_score90"][start:stop] = interval / scale[start:stop, None]
    return scores, output



@torch.no_grad()
def native_change_scores(knots, correlations, target, scale, pairs, *, count=4096,
                         seed=2026091517, batch_size=4):
    """Pairwise change CRPS using the common reconstructed marginal distributions."""
    return reconstructed_evidence(knots, correlations, target, scale, pairs,
        {"count": count, "seed": seed, "batch_size": batch_size})[0]


def calibrated_knots(quantiles, temperature):
    """Apply the shared temperature around each original forecast median."""
    center = quantiles[:, :, 4, None]
    return center + temperature * (quantiles.sort(-1).values - center)
