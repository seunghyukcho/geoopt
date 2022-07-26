import torch
from . import math
from ..base import Manifold
from torch.nn import functional as F


__all__ = ["DiagonalNormal"]


class DiagonalNormal(Manifold):
    ndim = 2
    reversible = False
    name = "DiagonalNormal"
    __scaling__ = Manifold.__scaling__.copy()

    def __init__(self):
        super().__init__()

    def _check_point_on_manifold(self, x: torch.Tensor, *, atol, rtol):
        ok = (x.size()[-1] == 2)
        reason = None
        if ok:
            ok = ((x[..., 1] <= 0) == 0)
            reason = None if ok else 'Standard deviation is not positive.'
        else:
            reason = 'Last dimension is not 2.'

        return ok, reason

    def _check_vector_on_tangent(self, x: torch.Tensor, u: torch.Tensor, *, atol, rtol):
        return True, True

    def inner(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, *, keepdim) -> torch.Tensor:
        metric_tensor = math.riemannian_metric(x)
        v = metric_tensor * v
        return (u * v).sum(dim=-1).sum(dim=-1, keepdim=keepdim)

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return u

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        x[..., 1] = F.softplus(x[..., 1])
        return x

    def dist(self, x: torch.Tensor, y: torch.Tensor, *, keepdim) -> torch.Tensor:
        return math.dist(x, y, keepdim)

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        metric_tensor = math.inv_riemannian_metric(x)
        u = u * metric_tensor
        return u

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return math.exp(x, u)

    retr = expmap

