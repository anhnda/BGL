"""
tier0_anytime_feasibility.py
============================
VIỆC #0 -- ANYTIME (time-uniform) FEASIBILITY CHECK for the LIME certificate.

THE QUESTION (single point of failure for both proposals).
Both the "Anytime-Valid Certification" and the "Certificate-Guided Adaptive
Sampling" proposals assume the *fixed-N* detection floor of this repo,

    F_N^fixed = C_est(Sigma_hat_N) * [ sigma_obs*sqrt(2L/N)
                                       + sqrt(2 m L/N)
                                       + (2/3) B L/N ],
    L = log(2 * split * pK / delta),
    C_est = max{ lambda_min(Sigma_hat)^{-1/2}, |||Sigma_hat^{-1}|||_inf },

can be turned into a *time-uniform* (anytime) floor F_t^CS valid simultaneously
for all t >= t0, WITHOUT the bound inflating so much that anytime stopping saves
nothing over the fixed budget. If F_t^CS / F_N^fixed is large, the whole LIME
branch of both papers loses its empirical punch. This script measures that
inflation BEFORE any theorem is committed to.

IT RUNS NOTHING ON A MODEL. Assumption 1 (design conditioning) depends on the
masks and basis only, not on rho or g_rho, so the design term is reference-free
and model-free (same philosophy as tier1b). The scalar query-noise / mismatch
terms are swept over (sigma_obs, m, B) as a-priori upper bounds, exactly as a
certificate must treat them.

WHY THE POPULATION TARGET IS KNOWN (the lever the proposal relies on).
Masks are Bernoulli(p_keep=0.5) i.i.d.; the centered Walsh design is +/-1 and
column-standardizes to itself, so the POPULATION Gram is the identity:

    Sigma = E[ X^T X / t ] = I_{pK},   lambda_min(Sigma) = 1 exactly,
    diag(Sigma_hat_t) = 1 exactly,
    off-diagonal (S,S') entry of Sigma_hat_t
        = (1/t) sum_s chi_{S Δ S'}(z_s)   -- a mean-zero bounded Rademacher mean.

So the design problem is a time-uniform *deviation of Sigma_hat_t around a known
I*, not around an unknown target. That is what lets us (i) use matrix-Bernstein
with the exact variance proxy, and (ii) separate the two pieces of C_est.

WHAT WE COMPUTE.
Three time-uniform building blocks, each reusing the repo's fixed-N structure so
the comparison is apples-to-apples:

  (S) SCALAR boundary for the noise and mismatch means. Two constructions:
      * epoch-stitch : union bound over geometric epochs of the EXACT bl two-term
                       Bernstein radius -- faithful to bl, slightly loose, and
                       obviously correct (reuses certified_floor term-for-term).
      * normal-mix   : Robbins normal-mixture confidence sequence tuned to an
                       anticipated horizon t_star (tighter; the "pay iterated-log
                       once" option of the proposal).

  (D) DESIGN constant C_design,t, a time-uniform upper bound on C_est(t),
      decomposed into its two parts because they behave very differently:
      * lambda_min^{-1/2} : controlled by time-uniform MATRIX-BERNSTEIN on
                            ||Sigma_hat_t - I||_op with the EXACT proxies
                            v = pK-1, R = pK+1 (derived below). Usable from
                            t0 ~ pK log pK -- cheap.
      * |||Sigma_hat^{-1}|||_inf : the max-abs-row-sum term. It has NO free lunch
                            under time-uniformity: the honest bounds are either
                            naive entrywise (t0 ~ pK^2 log pK, hopeless) or the
                            operator route sqrt(pK)/(1-dev_op) (usable early but
                            carrying a sqrt(pK) prefactor). This sqrt(pK) is the
                            real, irreducible inflation and the headline finding.

  (F) The assembled anytime floor and the INFLATION FACTOR F_t^CS / F_N^fixed,
      decomposed into {scalar iterated-log factor} x {design sqrt(pK) factor},
      plus the usable horizon t0 and the extra-sample cost tau_CS / N_fixed at a
      target resolution epsilon.

GPU. The optional `simulate` mode (torch, CUDA if available) draws R independent
mask banks, forms the batched Gram over a ladder of t, and (a) reports the
REALIZED C_est(N) and its lambda_min / inf-norm parts -- the honest fixed-N
denominator -- and (b) checks empirical coverage: along sample paths,
lambda_min(Sigma_hat_t) must stay >= 1 - dev_op(t) (validity sanity, not a
proof). The a-priori tables need only numpy.

Run (user runs this; this file never runs anything on import):
    python tier0_anytime_feasibility.py aprioi      # a-priori inflation tables (numpy)
    python tier0_anytime_feasibility.py horizon      # usable-horizon t0 table
    python tier0_anytime_feasibility.py cost          # extra-sample cost at target eps
    python tier0_anytime_feasibility.py simulate      # torch/GPU realized-Cest + coverage
    python tier0_anytime_feasibility.py all           # everything torch-free + note
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass

import numpy as np

import bl_core as bl


# =========================================================================== #
#  delta / log bookkeeping -- mirror bl_core exactly so the comparison is fair
# =========================================================================== #
def default_delta(d: int, K: int) -> float:
    """The paper's choice delta = 1/pK."""
    return 1.0 / bl.p_K(d, K)


def split_delta(delta: float, n_events: int) -> float:
    """Allocate the total failure budget across n_events certificate events.

    We keep the same top-level split as bl (design, noise, mismatch), and each
    time-uniform block then spends its share across its own epochs / mixture.
    """
    return delta / float(n_events)


# =========================================================================== #
#  (S) SCALAR time-uniform boundaries on a mean of bounded increments
# =========================================================================== #
def _bl_two_term_at(N: int, pK_log_L: float, sigma_obs: float, m: float,
                    B: float) -> float:
    """The bl two-term Bernstein radius (WITHOUT C_est) at a single N, using a
    pre-computed log factor. Mirrors certified_floor's bracket exactly:

        sigma_obs*sqrt(2L/N) + sqrt(2 m L/N) + (2/3) B L/N .
    """
    L = pK_log_L
    noise = sigma_obs * math.sqrt(2.0 * L / N)
    leak_g = math.sqrt(2.0 * max(m, 0.0) * L / N)
    leak_e = (2.0 / 3.0) * max(B, 0.0) * L / N
    return noise + leak_g + leak_e


def _log_L(d: int, K: int, delta: float) -> float:
    """L = log(2 * split * pK / delta) with split folded into delta already.

    Here we pass the per-event delta directly and use split=1 inside so the
    caller controls the budget. Equivalent to bl.log_pk_over_delta with the
    chosen delta and split=1.
    """
    return bl.log_pk_over_delta(d, K, delta=delta, split=1)


def n_epochs(t0: int, T: int, eta: float) -> int:
    """Number of geometric epochs [t0, t0*eta, ...] needed to cover [t0, T]."""
    if T <= t0:
        return 1
    return int(math.ceil(math.log(T / t0) / math.log(eta))) + 1


def epoch_lower(t: int, t0: int, eta: float) -> int:
    """Lower edge b_{k-1} of the geometric epoch containing t (t >= t0).

    On an epoch (b_{k-1}, b_k] the fixed-N radius (decreasing in N) is largest at
    the lower edge, so a bound using b_{k-1} is valid for every t in the epoch.
    """
    if t <= t0:
        return t0
    k = int(math.floor(math.log(t / t0) / math.log(eta)))
    return int(math.floor(t0 * (eta ** k)))


def scalar_stitch(t: int, d: int, K: int, *, sigma_obs: float, m: float,
                  B: float, delta_event: float, t0: int, T: int,
                  eta: float = 1.4) -> float:
    """Epoch-stitched time-uniform radius for the (noise + mismatch) mean.

    Union bound: allocate delta_event across the epochs with Robbins weights
    w_k ∝ 1/k^2 (sum pi^2/6), so the per-epoch level is delta_event * 6/(pi^2 k^2).
    Within the epoch we evaluate the EXACT bl two-term radius at the epoch's lower
    edge. This is faithful to the fixed-N certificate and manifestly valid.
    """
    if t < t0:
        return float("inf")
    Kep = n_epochs(t0, T, eta)
    norm = sum(1.0 / (k * k) for k in range(1, Kep + 1))
    # index k of the epoch containing t
    k = int(math.floor(math.log(max(t, t0) / t0) / math.log(eta))) + 1
    k = max(1, min(k, Kep))
    delta_k = delta_event * (1.0 / (k * k)) / norm
    L = _log_L(d, K, delta_k)
    b_lo = max(epoch_lower(t, t0, eta), 1)
    return _bl_two_term_at(b_lo, L, sigma_obs, m, B)


def scalar_normal_mixture(t: int, *, sigma: float, delta_event: float,
                          t_star: int) -> float:
    """Robbins two-sided normal-mixture confidence sequence on a mean.

    For sigma-sub-Gaussian increments with intrinsic time V_t = sigma^2 t and a
    tuning rho > 0, with probability >= 1 - delta_event simultaneously over t:

        |mean_t| <= (1/t) * sqrt( (V_t + rho) * log( (V_t + rho) / (rho * delta^2) ) ).

    rho is chosen rho = sigma^2 * t_star to minimize the radius near the
    anticipated certification horizon t_star (the "pay iterated-log once" knob).
    This covers the sub-Gaussian part only; for the mismatch sub-exponential tail
    use scalar_stitch, which carries the exact (m, B) Bernstein terms.
    """
    if sigma <= 0.0 or t <= 0:
        return 0.0
    V = sigma * sigma * t
    rho = sigma * sigma * max(t_star, 1)
    arg = (V + rho) / (rho * delta_event * delta_event)
    if arg <= 1.0:
        arg = 1.0 + 1e-12
    return math.sqrt((V + rho) * math.log(arg)) / t


# =========================================================================== #
#  (D) DESIGN constant: time-uniform upper bounds on C_est(t)
#
#  Per-sample matrix M_s = x_s x_s^T - I, x_s in {+-1}^pK (||x_s||^2 = pK).
#    * ||M_s||_op <= ||x_s||^2 + 1 = pK + 1                       =: R_mat
#    * (x x^T)^2 = ||x||^2 x x^T = pK x x^T  =>  E[M_s^2] = (pK-1) I
#      so the matrix variance proxy is  v = ||E[M_s^2]||_op = pK - 1 =: V_mat
#  Tropp matrix Bernstein (fixed t), two-sided, prob >= 1 - delta:
#    ||Sigma_hat_t - I||_op <= sqrt(2 v Lm / t) + (2/3) R Lm / t,
#    Lm = log(2 pK / delta).
#  Time-uniform version = epoch-stitch this over geometric epochs.
# =========================================================================== #
def dev_op_matrix_bernstein(t: int, d: int, K: int, *, delta_event: float,
                            t0: int, T: int, eta: float = 1.4) -> float:
    """Epoch-stitched time-uniform upper bound on ||Sigma_hat_t - I||_op."""
    pK = bl.p_K(d, K)
    if t < t0:
        return float("inf")
    Kep = n_epochs(t0, T, eta)
    norm = sum(1.0 / (k * k) for k in range(1, Kep + 1))
    k = int(math.floor(math.log(max(t, t0) / t0) / math.log(eta))) + 1
    k = max(1, min(k, Kep))
    delta_k = delta_event * (1.0 / (k * k)) / norm
    Lm = math.log(2.0 * pK / delta_k)
    b_lo = max(epoch_lower(t, t0, eta), 1)
    v = pK - 1.0
    R = pK + 1.0
    return math.sqrt(2.0 * v * Lm / b_lo) + (2.0 / 3.0) * R * Lm / b_lo


def dev_entry_hoeffding(t: int, d: int, K: int, *, delta_event: float,
                        t0: int, T: int, eta: float = 1.4) -> float:
    """Epoch-stitched time-uniform bound on the max |off-diagonal| entry of
    Sigma_hat_t - I (each entry a Rademacher mean, |.|<=1, var<=1), union over
    the n_off = pK(pK-1)/2 distinct off-diagonal entries via Hoeffding.

        e_t <= sqrt( log(2 n_off / delta') / (2 b_lo) ),  delta' per epoch.
    """
    pK = bl.p_K(d, K)
    n_off = pK * (pK - 1) // 2
    if n_off == 0 or t < t0:
        return 0.0 if n_off == 0 else float("inf")
    Kep = n_epochs(t0, T, eta)
    norm = sum(1.0 / (k * k) for k in range(1, Kep + 1))
    k = int(math.floor(math.log(max(t, t0) / t0) / math.log(eta))) + 1
    k = max(1, min(k, Kep))
    delta_k = delta_event * (1.0 / (k * k)) / norm
    b_lo = max(epoch_lower(t, t0, eta), 1)
    return math.sqrt(math.log(2.0 * n_off / delta_k) / (2.0 * b_lo))


@dataclass
class DesignConstant:
    """Time-uniform C_design,t decomposed into its two C_est parts.

    c_lmin   : upper bound on lambda_min(Sigma_hat_t)^{-1/2}   (matrix-Bernstein)
    c_inf_op : upper bound on |||Sigma_hat_t^{-1}|||_inf via the operator route
               sqrt(pK) / (1 - dev_op)                         (usable early)
    c_inf_naive : |||.|||_inf via naive entrywise row-sum       (hopeless t0)
    c_design : the honest max{ c_lmin, min(c_inf_op, c_inf_naive) }
    finite   : whether the bound is active (deviation < 1) at this t
    """
    c_lmin: float
    c_inf_op: float
    c_inf_naive: float
    c_design: float
    finite: bool


def design_constant(t: int, d: int, K: int, *, delta_event: float, t0: int,
                    T: int, eta: float = 1.4) -> DesignConstant:
    pK = bl.p_K(d, K)
    dev_op = dev_op_matrix_bernstein(t, d, K, delta_event=delta_event,
                                     t0=t0, T=T, eta=eta)
    e_t = dev_entry_hoeffding(t, d, K, delta_event=delta_event,
                              t0=t0, T=T, eta=eta)
    inf = float("inf")
    c_lmin = inf if dev_op >= 1.0 else (1.0 - dev_op) ** (-0.5)
    c_inf_op = inf if dev_op >= 1.0 else math.sqrt(pK) / (1.0 - dev_op)
    row_dev = (pK - 1.0) * e_t
    c_inf_naive = inf if row_dev >= 1.0 else 1.0 / (1.0 - row_dev)
    c_inf = min(c_inf_op, c_inf_naive)
    c_design = max(c_lmin, c_inf)
    return DesignConstant(c_lmin=c_lmin, c_inf_op=c_inf_op,
                          c_inf_naive=c_inf_naive, c_design=c_design,
                          finite=math.isfinite(c_design))


# =========================================================================== #
#  (F) Assembled anytime floor and the inflation factor
# =========================================================================== #
def anytime_floor(t: int, d: int, K: int, *, sigma_obs: float, m: float,
                  B: float, delta: float, t0: int, T: int, eta: float = 1.4,
                  scalar: str = "stitch", t_star: int = None) -> float:
    """F_t^CS = C_design,t * [ scalar time-uniform (noise+mismatch) radius ].

    Total delta split 3 ways (design, noise, mismatch), mirroring bl's split=3.
    """
    de = split_delta(delta, 3)
    dc = design_constant(t, d, K, delta_event=de, t0=t0, T=T, eta=eta)
    if not dc.finite:
        return float("inf")
    if scalar == "mix":
        ts = t_star if t_star is not None else max(t0, T // 4)
        noise = scalar_normal_mixture(t, sigma=sigma_obs, delta_event=de,
                                      t_star=ts)
        # mismatch sub-Gaussian part via mixture with sigma=sqrt(m); the
        # sub-exponential B/t tail is added from the faithful stitch term.
        mis_g = scalar_normal_mixture(t, sigma=math.sqrt(max(m, 0.0)),
                                      delta_event=de, t_star=ts)
        b_lo = max(epoch_lower(t, t0, eta), 1)
        Lm = _log_L(d, K, de)
        mis_e = (2.0 / 3.0) * max(B, 0.0) * Lm / b_lo
        bracket = noise + mis_g + mis_e
    else:
        bracket = scalar_stitch(t, d, K, sigma_obs=sigma_obs, m=m, B=B,
                                delta_event=de, t0=t0, T=T, eta=eta)
    return dc.c_design * bracket


def fixed_floor(N: int, d: int, K: int, *, c_est: float, sigma_obs: float,
                m: float, B: float, delta: float) -> float:
    """The repo's fixed-N certified floor at a given realized/assumed C_est."""
    return bl.certified_floor(c_est, sigma_obs, m, B, d=d, N=N, K=K,
                              delta=delta, split=3)


# =========================================================================== #
#  TABLES (a-priori, numpy only)
# =========================================================================== #
_GRID = [
    ("image   d=49 K=1", 49, 1),
    ("calib   d=30 K=1", 30, 1),
    ("nlp     d=13 K=1", 13, 1),
    ("enum    d=13 K=2", 13, 2),
    ("enum    d=18 K=2", 18, 2),
]


def _usable_t0(d: int, K: int, delta: float, T: int, eta: float,
               which: str) -> int:
    """Smallest t at which a design route becomes active (deviation < 1)."""
    de = split_delta(delta, 3)
    pK = bl.p_K(d, K)
    lo, hi = pK + 2, T
    # exponential search then bisection on the chosen route's finiteness
    def active(t):
        if which == "lmin" or which == "infop":
            return dev_op_matrix_bernstein(t, d, K, delta_event=de, t0=pK + 2,
                                           T=T, eta=eta) < 1.0
        return (pK - 1.0) * dev_entry_hoeffding(t, d, K, delta_event=de,
                                                t0=pK + 2, T=T, eta=eta) < 1.0
    t = lo
    while t < hi and not active(t):
        t *= 2
    if not active(min(t, hi)):
        return -1
    a, b = lo, min(t, hi)
    while b - a > 1:
        mdp = (a + b) // 2
        if active(mdp):
            b = mdp
        else:
            a = mdp
    return b


def table_horizon(delta_scale: float = 1.0, T: int = 2_000_000,
                  eta: float = 1.4):
    """Usable horizon t0 for each design route -- the 'does it even turn on'
    table. lmin via matrix-Bernstein should be ~pK log pK; inf-norm naive
    ~pK^2 log pK (shown to be hopeless)."""
    print("=" * 78)
    print("USABLE HORIZON t0 : smallest budget at which the anytime design term")
    print("is active (deviation bound < 1). Lower is better.")
    print("=" * 78)
    print(f"  {'point':>18} {'pK':>6} {'t0(lmin/op)':>13} {'t0(infnaive)':>13} "
          f"{'pK logpK':>10} {'pK^2 logpK':>12}")
    for name, d, K in _GRID:
        pK = bl.p_K(d, K)
        delta = default_delta(d, K) * delta_scale
        t_op = _usable_t0(d, K, delta, T, eta, "lmin")
        t_naive = _usable_t0(d, K, delta, T, eta, "entry")
        ref1 = pK * math.log(pK)
        ref2 = pK * pK * math.log(pK)
        f_op = f"{t_op:d}" if t_op > 0 else ">T"
        f_na = f"{t_naive:d}" if t_naive > 0 else ">T"
        print(f"  {name:>18} {pK:>6d} {f_op:>13} {f_na:>13} "
              f"{ref1:>10.0f} {ref2:>12.0f}")
    print("\n  READING: the matrix-Bernstein route (lambda_min and the operator")
    print("  inf-norm bound) turns on at t0 ~ pK log pK -- a feasible budget. The")
    print("  naive entrywise row-sum route needs ~pK^2 log pK and is hopeless at")
    print("  K=2. CONCLUSION: a usable anytime LIME certificate must control the")
    print("  Gram at the OPERATOR level, never entrywise.")


def table_inflation(N_list=(512, 1000, 2000, 4000, 8000, 16000),
                    sigma_obs: float = 0.05, m: float = 0.05, B: float = 1.0,
                    delta_scale: float = 1.0, T: int = 2_000_000,
                    eta: float = 1.4, c_est_fixed: float = 2.0,
                    scalar: str = "stitch"):
    """Headline inflation F_t^CS / F_N^fixed at matched t = N, decomposed into
    the scalar (iterated-log) factor and the design (sqrt(pK)) factor.

    c_est_fixed is the realized fixed-N C_est to use in the denominator; the
    `simulate` mode measures it (typically ~1.5-3.8 per the paper). Pass the
    measured value here for the honest comparison; the default 2.0 is a
    placeholder in the paper's reported band.
    """
    print("=" * 78)
    print(f"INFLATION  F_t^CS / F_N^fixed  at t=N   (scalar={scalar}, "
          f"sigma={sigma_obs}, m={m}, B={B})")
    print(f"  fixed-N denominator uses realized C_est = {c_est_fixed} "
          f"(measure via `simulate`)")
    print("=" * 78)
    for name, d, K in _GRID:
        pK = bl.p_K(d, K)
        delta = default_delta(d, K) * delta_scale
        print(f"\n  [{name}]  pK={pK}  sqrt(pK)={math.sqrt(pK):.2f}")
        print(f"    {'N':>7} {'F_fixed':>11} {'F_CS':>11} {'inflate':>8} "
              f"{'=scalarx':>9} {'x designx':>10} {'C_design':>9}")
        de = split_delta(delta, 3)
        for N in N_list:
            if N <= pK:
                print(f"    {N:>7d} {'--':>11} {'--':>11} "
                      f"{'infeasible (N<=pK)':>30}")
                continue
            ff = fixed_floor(N, d, K, c_est=c_est_fixed, sigma_obs=sigma_obs,
                             m=m, B=B, delta=delta)
            fcs = anytime_floor(N, d, K, sigma_obs=sigma_obs, m=m, B=B,
                                delta=delta, t0=pK + 2, T=T, eta=eta,
                                scalar=scalar)
            dc = design_constant(N, d, K, delta_event=de, t0=pK + 2, T=T,
                                 eta=eta)
            if not math.isfinite(fcs):
                print(f"    {N:>7d} {ff:>11.5f} {'inf':>11} "
                      f"{'design term not active yet':>30}")
                continue
            infl = fcs / ff if ff > 0 else float("inf")
            # decomposition: design factor vs c_est_fixed, scalar factor = rest
            design_x = dc.c_design / c_est_fixed
            scalar_x = infl / design_x if design_x > 0 else float("nan")
            print(f"    {N:>7d} {ff:>11.5f} {fcs:>11.5f} {infl:>8.2f} "
                  f"{scalar_x:>9.2f} {design_x:>10.2f} {dc.c_design:>9.2f}")
    print("\n  READING: inflation factorizes as (scalar iterated-log) x (design).")
    print("  The scalar factor is modest and shrinks with N (epoch-stitch pays a")
    print("  slowly-growing union-bound penalty). The DESIGN factor is dominated")
    print("  by the inf-norm term's sqrt(pK) prefactor -- the irreducible cost of")
    print("  carrying |||Sigma_hat^{-1}|||_inf anytime. ACTION: if sqrt(pK) is too")
    print("  large, replace C_est*||.||_inf by a self-normalized (Mahalanobis)")
    print("  radius in the certificate, which removes the row-sum term entirely.")


def table_cost(sigma_obs: float = 0.05, m: float = 0.05, B: float = 1.0,
               eps_list=(0.05, 0.02, 0.01), delta_scale: float = 1.0,
               T: int = 2_000_000, eta: float = 1.4, c_est_fixed: float = 2.0,
               scalar: str = "stitch"):
    """Extra-sample cost: for target resolution eps, N_fixed(eps) vs tau_CS(eps)
    and the ratio tau_CS / N_fixed (how many more queries anytime stopping pays
    for the right to stop whenever it likes)."""
    print("=" * 78)
    print(f"EXTRA-SAMPLE COST  tau_CS / N_fixed  at target resolution eps "
          f"(scalar={scalar})")
    print(f"  (how much the right to stop at ANY data-chosen time costs vs a "
          f"pre-set N)")
    print("=" * 78)

    def first_below(fn, eps):
        t = 8
        while t < T and not (math.isfinite(fn(t)) and fn(t) <= eps):
            t = int(t * 1.3) + 1
        return t if t < T else -1

    for name, d, K in _GRID:
        pK = bl.p_K(d, K)
        delta = default_delta(d, K) * delta_scale
        print(f"\n  [{name}]  pK={pK}")
        print(f"    {'eps':>7} {'N_fixed':>9} {'tau_CS':>9} {'ratio':>7}")
        ffun = lambda N: fixed_floor(N, d, K, c_est=c_est_fixed,
                                     sigma_obs=sigma_obs, m=m, B=B, delta=delta)
        cfun = lambda t: anytime_floor(t, d, K, sigma_obs=sigma_obs, m=m, B=B,
                                       delta=delta, t0=pK + 2, T=T, eta=eta,
                                       scalar=scalar)
        for eps in eps_list:
            Nf = first_below(ffun, eps)
            tc = first_below(cfun, eps)
            if Nf < 0 or tc < 0:
                print(f"    {eps:>7.3f} {('>T' if Nf<0 else Nf):>9} "
                      f"{('>T' if tc<0 else tc):>9} {'--':>7}")
                continue
            print(f"    {eps:>7.3f} {Nf:>9d} {tc:>9d} {tc / Nf:>7.2f}")
    print("\n  READING: the ratio is the multiplicative query premium of an")
    print("  anytime-valid stop. A premium near 1.x-2.x is the regime where the")
    print("  adaptive proposal's stopping story survives; a premium of 5x+ means")
    print("  the LIME certificate must be reformulated (self-normalized radius)")
    print("  before the empirical claims can hold.")


# =========================================================================== #
#  SIMULATE (torch, GPU) -- realized C_est denominator + coverage sanity
# =========================================================================== #
def _torch_device(pref: str):
    import torch
    if pref == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("  [note] CUDA not available; falling back to CPU for simulate.")
    return torch.device("cpu")


def _features_torch(Z, K, torch):
    """Augmented +/-1 Walsh design [1, chi_i, chi_i chi_j] for a batch of masks.

    Z: (R, t, d) in {0,1}. Returns (R, t, pK). Columns standardize to themselves
    (all +/-1), so this IS the standardized design bl_core uses.
    """
    Zc = 2.0 * (Z - 0.5)                       # (R, t, d) in {-1,+1}
    R, t, d = Zc.shape
    ones = torch.ones((R, t, 1), dtype=Zc.dtype, device=Zc.device)
    cols = [ones, Zc]
    if K >= 2:
        idx = torch.combinations(torch.arange(d, device=Zc.device), 2)
        pair = Zc[:, :, idx[:, 0]] * Zc[:, :, idx[:, 1]]   # (R, t, C(d,2))
        cols.append(pair)
    return torch.cat(cols, dim=2)


def simulate(d: int = 49, K: int = 1, N_list=(512, 1000, 2000, 4000, 8000),
             R: int = 64, seed0: int = 0, device_pref: str = "cuda",
             delta_scale: float = 1.0, T: int = 2_000_000, eta: float = 1.4):
    """Draw R mask banks on GPU; report realized C_est(N) and its two parts, and
    check that lambda_min(Sigma_hat_t) >= 1 - dev_op(t) along the ladder
    (empirical coverage of the time-uniform design bound).

    The realized C_est here is exactly the denominator the inflation table needs:
    feed its mean back via --c_est.
    """
    try:
        import torch
    except Exception as exc:                       # pragma: no cover
        print(f"  torch unavailable ({exc}); simulate needs torch. "
              f"Use the numpy a-priori tables instead.")
        return
    dev = _torch_device(device_pref)
    pK = bl.p_K(d, K)
    delta = default_delta(d, K) * delta_scale
    de = split_delta(delta, 3)
    Nmax = max(N_list)
    g = torch.Generator(device="cpu").manual_seed(seed0)
    # draw on CPU then move (Bernoulli(0.5)); keep memory modest
    Z = (torch.rand((R, Nmax, d), generator=g) > 0.5).to(torch.float64).to(dev)
    X = _features_torch(Z, K, torch)              # (R, Nmax, pK)
    Ipk = torch.eye(pK, dtype=torch.float64, device=dev)

    print("=" * 78)
    print(f"SIMULATE (torch/{dev.type})  d={d} K={K} pK={pK} R={R}")
    print("  realized C_est(N) and its parts; coverage of the anytime design "
          "bound")
    print("=" * 78)
    print(f"  {'N':>7} {'lmin':>8} {'infnorm':>9} {'C_est':>8} "
          f"{'dev_op':>8} {'bound lmin>=':>12} {'cover?':>7}")
    cests = {}
    worst_cover = True
    for N in N_list:
        if N <= pK:
            print(f"  {N:>7d} {'--':>8} {'--':>9} {'--':>8} "
                  f"{'infeasible':>30}")
            continue
        Xn = X[:, :N, :]                           # (R, N, pK)
        G = torch.matmul(Xn.transpose(1, 2), Xn) / N   # (R, pK, pK)
        evals = torch.linalg.eigvalsh(G)           # (R, pK)
        lmin = evals[:, 0]
        Ginv = torch.linalg.inv(G)
        inf_norm = torch.abs(Ginv).sum(dim=2).amax(dim=1)   # max abs row sum
        cest = torch.maximum(lmin.clamp(min=1e-12) ** (-0.5), inf_norm)
        dev_op_bound = dev_op_matrix_bernstein(N, d, K, delta_event=de,
                                               t0=pK + 2, T=T, eta=eta)
        lmin_floor = 1.0 - dev_op_bound
        covered = bool((lmin >= lmin_floor - 1e-9).all().item()) \
            if math.isfinite(lmin_floor) else True
        worst_cover = worst_cover and covered
        cests[N] = float(cest.mean().item())
        print(f"  {N:>7d} {lmin.mean().item():>8.4f} "
              f"{inf_norm.mean().item():>9.4f} {cest.mean().item():>8.4f} "
              f"{dev_op_bound:>8.4f} "
              f"{(lmin_floor if math.isfinite(lmin_floor) else float('nan')):>12.4f} "
              f"{('yes' if covered else 'NO'):>7}")
    print("\n  READING: realized C_est is O(1)-O(few) (the honest fixed-N")
    print("  denominator -- feed its mean to the inflation table via --c_est).")
    print("  'cover? = yes' on every rung means the time-uniform lambda_min bound")
    print("  was never violated on these paths (a validity sanity check, not a")
    print(f"  proof). Overall coverage held: {worst_cover}.")
    if cests:
        mean_cest = float(np.mean(list(cests.values())))
        print(f"\n  suggested --c_est {mean_cest:.3f}  (mean realized C_est)")
    return cests


# =========================================================================== #
def build_parser():
    p = argparse.ArgumentParser(
        description="Việc #0 anytime-feasibility check for the LIME certificate")
    p.add_argument("mode", nargs="?", default="all",
                   choices=["aprioi", "apriori", "inflation", "horizon",
                            "cost", "simulate", "all"],
                   help="which analysis to run")
    p.add_argument("--sigma_obs", type=float, default=0.05)
    p.add_argument("--m", type=float, default=0.05, help="mismatch energy bound")
    p.add_argument("--B", type=float, default=1.0, help="mismatch sup-norm bound")
    p.add_argument("--c_est", type=float, default=2.0,
                   help="realized fixed-N C_est for the denominator "
                        "(measure via `simulate`)")
    p.add_argument("--scalar", choices=["stitch", "mix"], default="stitch")
    p.add_argument("--eta", type=float, default=1.4, help="epoch growth factor")
    p.add_argument("--T", type=int, default=2_000_000, help="horizon cap")
    p.add_argument("--delta_scale", type=float, default=1.0)
    # simulate-only
    p.add_argument("--d", type=int, default=49)
    p.add_argument("--K", type=int, default=1)
    p.add_argument("--R", type=int, default=64, help="independent mask banks")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    mode = "apriori" if args.mode == "aprioi" else args.mode

    def run_apriori():
        table_inflation(sigma_obs=args.sigma_obs, m=args.m, B=args.B,
                        delta_scale=args.delta_scale, T=args.T, eta=args.eta,
                        c_est_fixed=args.c_est, scalar=args.scalar)

    if mode in ("apriori", "inflation"):
        run_apriori()
    elif mode == "horizon":
        table_horizon(delta_scale=args.delta_scale, T=args.T, eta=args.eta)
    elif mode == "cost":
        table_cost(sigma_obs=args.sigma_obs, m=args.m, B=args.B,
                   delta_scale=args.delta_scale, T=args.T, eta=args.eta,
                   c_est_fixed=args.c_est, scalar=args.scalar)
    elif mode == "simulate":
        simulate(d=args.d, K=args.K, R=args.R, seed0=args.seed,
                 device_pref=args.device, delta_scale=args.delta_scale,
                 T=args.T, eta=args.eta)
    else:  # all (torch-free): horizon + inflation + cost, then a pointer
        table_horizon(delta_scale=args.delta_scale, T=args.T, eta=args.eta)
        print()
        run_apriori()
        print()
        table_cost(sigma_obs=args.sigma_obs, m=args.m, B=args.B,
                   delta_scale=args.delta_scale, T=args.T, eta=args.eta,
                   c_est_fixed=args.c_est, scalar=args.scalar)
        print("\n" + "=" * 78)
        print("NEXT: run  `python tier0_anytime_feasibility.py simulate "
              "--device cuda`\nto measure the realized C_est denominator on GPU, "
              "then re-run the\ninflation table with  --c_est <measured mean>.")
        print("=" * 78)


if __name__ == "__main__":
    main()
