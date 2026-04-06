"""SIGReg: Sketched-Isotropic-Gaussian Regularizer.

Anti-collapse regularizer from LeWM (Maes et al., 2026). Projects embeddings
onto M random unit-norm directions and checks normality via Epps-Pulley test
statistic. By Cramer-Wold theorem, matching all 1D marginals is equivalent
to matching the full joint N(0, I) distribution.

Ref: Balestriero & LeCun 2025, "Provable and scalable self-supervised
learning without the heuristics" (arXiv:2511.08544).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sigreg(
    Z: torch.Tensor,
    M: int = 1024,
    n_nodes: int = 50,
    t_min: float = 0.2,
    t_max: float = 4.0,
    lam: float = 2.0,
) -> torch.Tensor:
    """Compute SIGReg loss on embeddings.

    Minimized when Z ~ N(0, I). Projects Z onto M random directions,
    computes the Epps-Pulley normality test statistic on each 1D
    projection, and averages.

    Args:
        Z: (N, D) embedding matrix. N samples, D dimensions.
        M: number of random unit-norm projection directions.
        n_nodes: quadrature nodes for Epps-Pulley integral.
        t_min, t_max: integration range for characteristic function.
        lam: bandwidth of Gaussian weighting function w(t) = exp(-t^2/(2*lam^2)).

    Returns:
        Scalar loss (differentiable).
    """
    N, D = Z.shape

    # Random unit-norm projection directions
    U = torch.randn(D, M, device=Z.device, dtype=Z.dtype)
    U = F.normalize(U, dim=0)  # (D, M)

    # Project: H[n, m] = Z[n, :] . U[:, m]
    H = Z @ U  # (N, M)

    # Quadrature nodes for trapezoid integration
    t_nodes = torch.linspace(t_min, t_max, n_nodes, device=Z.device, dtype=Z.dtype)

    # Gaussian weighting: w(t) = exp(-t^2 / (2 * lam^2))
    w = torch.exp(-t_nodes ** 2 / (2 * lam ** 2))  # (n_nodes,)

    # Compute |phi_N(t) - phi_0(t)|^2 for each (t, m)
    # phi_N(t; h) = (1/N) sum_n exp(i*t*h_n) [empirical characteristic function]
    # phi_0(t) = exp(-t^2/2) [standard Gaussian CF]
    phase = t_nodes[:, None, None] * H[None, :, :]  # (n_nodes, N, M)

    ecf_real = torch.cos(phase).mean(dim=1)  # (n_nodes, M)
    ecf_imag = torch.sin(phase).mean(dim=1)  # (n_nodes, M)

    target_cf = torch.exp(-t_nodes ** 2 / 2)  # (n_nodes,)

    # |phi_N - phi_0|^2 = (Re(phi_N) - phi_0)^2 + Im(phi_N)^2
    diff_sq = (ecf_real - target_cf[:, None]) ** 2 + ecf_imag ** 2  # (n_nodes, M)

    # Weighted integral via trapezoid, then average over projections
    integrand = w[:, None] * diff_sq  # (n_nodes, M)
    integral = torch.trapezoid(integrand, t_nodes, dim=0)  # (M,)

    return integral.mean()
