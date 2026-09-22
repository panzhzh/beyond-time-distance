"""Differentiable interpolation of fixed-marginal CRPS along a contraction path.

Only the training objective is interpolated. Prediction remains the
PSD rational kernel, and selected heads are scored by the original quadrature.
PCHIP slopes follow the Fritsch--Butland rule documented by SciPy:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html
"""
from dataclasses import dataclass

import torch


@dataclass
class ContractionRisk:
    grid: torch.Tensor
    values: torch.Tensor
    slopes: torch.Tensor

    @classmethod
    def create(cls, grid, values):
        if grid.ndim != 1 or len(grid) < 3 or values.shape[-1] != len(grid):
            raise ValueError('Expected at least three grid points on the final axis')
        if not bool((grid.diff() > 0).all()) or not bool(torch.isfinite(values).all()):
            raise ValueError('Grid must increase and risks must be finite')
        h = grid.diff()
        delta = values.diff(dim=-1) / h
        before, after = delta[..., :-1], delta[..., 1:]
        same = before * after > 0
        w1, w2 = 2*h[1:]+h[:-1], h[1:]+2*h[:-1]
        safe_before = torch.where(same, before, torch.ones_like(before))
        safe_after = torch.where(same, after, torch.ones_like(after))
        middle = torch.where(same, (w1+w2)/(w1/safe_before+w2/safe_after), 0.)

        def edge(h0, h1, d0, d1):
            slope = ((2*h0+h1)*d0-h0*d1)/(h0+h1)
            slope = torch.where(slope.sign() != d0.sign(), 0., slope)
            return torch.where((d0.sign() != d1.sign()) & (slope.abs() > 3*d0.abs()), 3*d0, slope)

        first = edge(h[0], h[1], delta[..., 0], delta[..., 1])
        last = edge(h[-1], h[-2], delta[..., -1], delta[..., -2])
        slopes = torch.cat([first[..., None], middle, last[..., None]], dim=-1)
        return cls(grid, values, slopes)

    def __call__(self, contraction):
        if contraction.shape != self.values.shape[:-1]:
            raise ValueError('One contraction value is required per stored pair')
        x = contraction.clamp(self.grid[0], self.grid[-1])
        ix = torch.searchsorted(self.grid, x.contiguous(), right=True).sub(1).clamp(0, len(self.grid)-2)
        h = self.grid[ix+1]-self.grid[ix]
        t = (x-self.grid[ix])/h
        take = lambda a, j: a.gather(-1, j[..., None]).squeeze(-1)
        y0, y1 = take(self.values, ix), take(self.values, ix+1)
        m0, m1 = take(self.slopes, ix), take(self.slopes, ix+1)
        return ((2*t**3-3*t**2+1)*y0+(t**3-2*t**2+t)*h*m0
                +(-2*t**3+3*t**2)*y1+(t**3-t**2)*h*m1)
