"""
bl_core.py
==========
Single shared numerical core for the finite-budget LIME certification
experiments, rewritten to the consolidated experimental design.

DESIGN PRINCIPLE (one inequality, two directions).
The paper proves exactly one bound (Theorem 1):

    ||beta_hat - beta||_inf  <=  floor(N, rho)
    floor(N, rho) = Cest * sigma_eff * sqrt(2 log pK / N)
    sigma_eff     = sigma_obs + Cm * sqrt(m>K,rho)

Every experiment is a reading of this in ONE of two directions:
  * FORWARD  (the guarantee): |beta_hat_S| > floor  =>  sign is correct.
  * BACKWARD (the budget rule, Eq. 8): N >~ 2 Cest^2 sigma_eff^2 log pK / beta_min^2.
Anything that is not a forward sign check or a backward budget check
(set-nesting, count-monotonicity) is a WORKFLOW DIAGNOSTIC, never theorem
evidence, and is reported separately.

SINGLE SOURCE OF CONSTANTS.
The old codebase floated three constants (Cest=1 theoretical, 1.81 synthetic,
C_budget=3.5). Here there are exactly two, defined once in CONSTANTS, calibrated
on synthetic data (Tier 1) and FROZEN everywhere downstream. They are never
re-fit on real data. The two roles of the estimator constant are kept explicit
and separate:
  * C_FLOOR  -- the constant in the floor BOUND, used forward.
  * C_BUDGET -- the constant that makes the budget rule LAND at the target,
                used backward (back-solved from the signed-detection transition).
Using C_FLOOR to invert the budget produces N_pred below the feasibility floor
~pK and is physically meaningless; the two constants are therefore distinct
objects, related by the conservative direction realized_floor <= beta_min.

Pure numpy. No torch. The black-box model wrappers live in the driver files and
call into this module; this module never imports torch.

REVISION (correctness fixes R1.2, R1.5, R1.6).
Three referee correctness points are implemented here, at the single source of
truth, so every tier inherits them without re-deriving anything:

  R1.2  delta-budget accounting.  Theorem 1 was stated at 1 - delta while its
        proof consumes several 1 - delta events (design conditioning, the
        sub-Gaussian query-noise maximum, and the mismatch-leakage bound of
        Lemma 1).  We split delta by a union bound over these events.  The floor
        therefore carries log(SPLIT * pK / delta) rather than log(pK / delta),
        computed once in log_pk_over_delta() and routed through the floor, the
        budget rule, and every downstream comparator.  With delta = 1/pK this
        turns the old 2 log pK into 2 log pK + 2 log SPLIT, a bounded additive
        correction that vanishes as pK grows.  The pilot scale is handled as an
        explicit conditioning hypothesis (Theorem 1's "conditional on sigma_eff
        valid"), i.e. SPLIT counts the three probabilistic events; set
        DELTA_SPLIT = 4 to also spend a share on the pilot event.

  R1.5  sub-exponential Bernstein term in Lemma 1.  The leakage bound is
        Bernstein, not purely sub-Gaussian: eta_N <= sqrt(2 m log(.)/N) +
        (2/3) B log(.)/N with B = ||r>K||_inf.  We now compute BOTH terms
        (leakage_bound_terms) and the regime N >~ (B^2/m) log(.) in which the
        sub-exponential tail is dominated and absorbed into C_M.  Tier 1 checks
        that its calibration cells sit inside that regime.

  R1.6  negative mismatch estimate + clipping bias.  The held-out mismatch
        estimate is a difference of variances and can be negative.  We return
        the RAW value and a flag alongside the clipped value, so callers can log
        the fraction of pilot draws with m_hat < 0.  Clipping at zero moves
        sigma_eff UP (conservative); the raw value is retained only for
        reporting, never fed to the floor.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from itertools import combinations
import numpy as np


# =========================================================================== #
#  CONSTANTS -- the single source of truth (calibrated at Tier 1, frozen here)
# =========================================================================== #
@dataclass(frozen=True)
class Constants:
    """The two and only floor constants, plus the design's fixed parameters.

    C_FLOOR and C_M enter the floor BOUND (forward direction). C_BUDGET enters
    the budget RULE (backward direction). C_FLOOR's theoretical value is 1 for
    the orthonormal +-1 design; the empirical value is expected to be >= 1
    because the bound is an upper bound under finite N, query noise, and
    mismatch -- this gap is not an anomaly and is stated once, here.

    R1.3 / R1.5 -- NORMALIZER PROVENANCE (why C_M = 0.833, not 1.24). The referee
    noted that the pre-revision C_M was calibrated against sqrt(m log pK / N)
    while the downstream floor carried a different factor, so consistency could
    not be read off. We resolve this by making ONE log factor canonical
    everywhere: Lemma 1's normalizer, the floor, and the C_M calibration all use
    log_pk_over_delta = log(DELTA_SPLIT * pK / delta) with delta = 1/pK (R1.2).
    Two effects move the constant, both benign:
      (i)  the larger normalizer rescales the ratio down (this alone would give
           ~0.814, whose product C_M*sqrt(normalizer) matches the old 1.24 to
           0.01% -- a pure re-expression of the SAME bound);
      (ii) C_M is an ASYMPTOTIC sqrt-regime quantity, so it is frozen from the
           large-N rows (N >= 2000) rather than the all-N mean. This lifts it to
           0.833. The ~2.4% difference from (i) is exactly the R1.5 sub-exp
           inflation that contaminates small-N rows and that the large-N freeze
           removes by construction (quantified per cell in Tier 1's `subexp`
           column). C_M = 0.833 is therefore the leakage constant in the clean
           sqrt regime, consistent with the floor's normalizer.
    C_BUDGET is back-solved (Tier 1 backward) against the same factor, giving
    1.535 (was 1.81).

    R1.4 -- Cest IS NO LONGER A FROZEN CONSTANT (Path A). The Tier-1b transfer
    study shows the forward floor constant Cest = max{gamma^{-1/2},
    |||Ginv|||_inf} does NOT collapse in pK/N: at fixed ratio it still splits by
    pK (and thus by d and K), 40-58% within-bin, and the empirical value at the
    deployed points is 1.5-3.8x, not 1.0. A single frozen scalar is therefore
    provably wrong. We instead MEASURE Cest per run from the realized design
    (realized_cest / floor_from_design): it is a pure model-free design quantity
    computable in milliseconds from the same Gram the OLS fit uses. C_FLOOR is
    retained ONLY as the orthonormal ideal (=1) and as a fallback when the Gram
    is too ill-conditioned to invert (in which case the fit itself fails). The
    forward floor used for every certified decision is floor_from_design(...),
    which needs no transfer claim because nothing is transferred.
    """
    C_FLOOR: float = 1.0       # orthonormal IDEAL / fallback only -- see R1.4 note
    C_M: float = 0.833         # leakage constant (Lemma 1); see R1.3/R1.5 note
    C_BUDGET: float = 1.535    # budget-rule constant, back-solved at Tier 1 (R1.2)
    P_KEEP: float = 0.5        # centered +-1 Walsh design the floor assumes
    Z_ALPHA: float = 1.96      # single pre-registered coord (two-sided 95%)
    # R1.2: number of 1 - delta events the union bound splits delta over.
    # 3 = {design conditioning (Assump. 1), query-noise sub-Gaussian maximum,
    # mismatch leakage (Lemma 1)}; the pilot scale is a stated CONDITIONING
    # hypothesis (Theorem 1), not a spent event. Set to 4 to also budget the
    # pilot event probabilistically. The floor carries log(DELTA_SPLIT*pK/delta).
    DELTA_SPLIT: int = 3


CONSTANTS = Constants()


# =========================================================================== #
#  Degree-K feature machinery  (identical across every setting and tier)
# =========================================================================== #
def p_K(d: int, K: int = 1) -> int:
    """Candidate coefficient count INCLUDING the intercept (paper's pK)."""
    if K == 1:
        return d + 1
    if K == 2:
        return 1 + d + d * (d - 1) // 2
    raise ValueError("only K in {1, 2} supported")


def log_pk_over_delta(d: int, K: int = 1, delta: float = None,
                      split: int = None) -> float:
    """R1.2 -- the union-bounded log factor that the floor carries.

    Theorem 1's proof is a union bound over `split` failure events, so the honest
    per-event level is delta/split and the concentration log is

        log( split * pK / delta ).

    The experiments use delta = 1/pK (the paper's choice), giving

        log( split * pK^2 )  =  2 log pK + log split,

    which recovers the old `2 log pK` PLUS the bounded correction `log split`.
    Passing split=1 and delta=1/pK reproduces the pre-revision factor exactly, so
    the change is auditable against the old numbers.
    """
    pk = p_K(d, K)
    delta = (1.0 / pk) if delta is None else delta
    split = CONSTANTS.DELTA_SPLIT if split is None else split
    return math.log(split * pk / delta)


def feature_subsets(d: int, K: int):
    """Ordered non-empty subsets |S| <= K: singletons then pairs (i<j)."""
    subs = [(i,) for i in range(d)]
    if K >= 2:
        subs += list(combinations(range(d), 2))
    return subs


def design_matrix(Z: np.ndarray, K: int) -> np.ndarray:
    """Centered Walsh design over |S| <= K, no intercept column (centering of y
    absorbs it). Main effects chi_i = 2(z_i - 1/2) in {-1,+1}; pair columns are
    products of +-1 columns (hence also +-1). Returns (N, pK-1)."""
    Zc = 2.0 * (Z - 0.5)
    N, d = Zc.shape
    if K == 1:
        return Zc
    pair_cols = [Zc[:, i] * Zc[:, j] for (i, j) in combinations(range(d), 2)]
    if pair_cols:
        return np.concatenate([Zc, np.stack(pair_cols, axis=1)], axis=1)
    return Zc


def standardize_columns(X: np.ndarray):
    scale = np.sqrt((X ** 2).mean(axis=0))
    scale = np.where(scale > 0, scale, 1.0)
    return X / scale, scale


def sample_masks(N: int, d: int, rng: np.random.Generator,
                 p_keep: float = None) -> np.ndarray:
    p_keep = CONSTANTS.P_KEEP if p_keep is None else p_keep
    return (rng.random((N, d)) > (1.0 - p_keep)).astype(float)


# =========================================================================== #
#  Dense OLS  (the paper's estimator; closed-form normal equations)
# =========================================================================== #
def ols_fit(Z: np.ndarray, y: np.ndarray, K: int):
    """Column-standardized dense OLS with intercept over the degree-<=K design.

    Returns (beta on ORIGINAL scale, intercept, diag(Ginv)). Raises
    LinAlgError if N <= pK (dense fit not well-posed -- Assumption 1).
    """
    d = Z.shape[1]
    X = design_matrix(Z, K)
    N = X.shape[0]
    if N <= p_K(d, K):
        raise np.linalg.LinAlgError(
            f"N={N} <= pK={p_K(d, K)}: dense K={K} fit not well-posed")
    Xs, scale = standardize_columns(X)
    y_mean = y.mean()
    G = (Xs.T @ Xs) / N
    if np.linalg.cond(G) > 1e8:
        raise np.linalg.LinAlgError("Gram ill-conditioned (N too small)")
    Ginv = np.linalg.inv(G)
    beta_std = Ginv @ (Xs.T @ (y - y_mean)) / N
    return beta_std / scale, y_mean, np.diag(Ginv)


# =========================================================================== #
#  R1.4 -- design-conditioning probe: the empirical Cest and its ingredients
#          as functions of the coordinate-to-budget ratio pK/N.
#
#  The referee's most serious point: Cest = max{gamma^{-1/2}, |||Ginv|||_inf}
#  is a max ABSOLUTE ROW SUM of the inverse empirical Gram, which grows with the
#  number of fitted coordinates pK at fixed budget N. It is calibrated at
#  (d=30, K=1, pK=31) and then FROZEN and transferred to d=49 and to K=2 designs
#  with far more coordinates. This probe measures gamma = lambda_min(Sigma_hat),
#  Cinv = |||Sigma_hat^{-1}|||_inf, and Cest = max{gamma^{-1/2}, Cinv} DIRECTLY
#  from a sampled Walsh design, so Tier 1 can chart Cest vs pK/N and test whether
#  the frozen value is stable in that ratio (the transfer claim) rather than in
#  d or K separately.
# =========================================================================== #
def op_inf_norm(M: np.ndarray) -> float:
    """||| M |||_inf = max_i sum_j |M_ij|  (the ell_inf -> ell_inf operator
    norm, i.e. the maximum absolute row sum)."""
    return float(np.max(np.abs(M).sum(axis=1)))


@dataclass
class ConditioningProbe:
    """R1.4 design-conditioning measurement at one (d, K, N) point.

    gamma     : lambda_min(Sigma_hat)         (Assumption 1 lower bound)
    Cinv      : |||Sigma_hat^{-1}|||_inf       (max abs row sum -- grows with pK)
    Cest_emp  : max{gamma^{-1/2}, Cinv}        (the empirical floor constant)
    ratio     : pK / N                         (the transfer axis)
    well_posed: whether the Gram was invertible at this (N, pK)
    """
    d: int
    K: int
    N: int
    pK: int
    ratio: float
    gamma: float
    Cinv: float
    Cest_emp: float
    well_posed: bool


def measure_conditioning(d: int, K: int, N: int, rng: np.random.Generator
                         ) -> ConditioningProbe:
    """Sample N Walsh masks, form the standardized degree-K Gram, and read off
    gamma, Cinv and Cest_emp. Pure design measurement: no response, no signal --
    Assumption 1 concerns the design only (the columns depend on masks/basis, not
    on rho or g_rho), which is exactly why this is reference-free.
    """
    pk = p_K(d, K)
    Z = sample_masks(N, d, rng)
    X = design_matrix(Z, K)
    Xs, _ = standardize_columns(X)
    G = (Xs.T @ Xs) / N
    ratio = pk / N
    try:
        if np.linalg.cond(G) > 1e10:
            raise np.linalg.LinAlgError
        Ginv = np.linalg.inv(G)
        gamma = float(np.linalg.eigvalsh(G)[0])       # lambda_min
        Cinv = op_inf_norm(Ginv)
        gpow = gamma ** (-0.5) if gamma > 0 else float("inf")
        Cest_emp = max(gpow, Cinv)
        wp = math.isfinite(Cest_emp)
    except np.linalg.LinAlgError:
        gamma, Cinv, Cest_emp, wp = float("nan"), float("nan"), float("nan"), False
    return ConditioningProbe(d=d, K=K, N=N, pK=pk, ratio=ratio, gamma=gamma,
                             Cinv=Cinv, Cest_emp=Cest_emp, well_posed=wp)


def realized_cest(Z: np.ndarray, K: int) -> float:
    """R1.4 (Path A) -- the forward floor constant Cest read off the ACTUAL
    design that will be used for the fit, rather than a frozen scalar.

    Cest = max{ gamma^{-1/2}, |||Sigma_hat^{-1}|||_inf }  on the standardized
    degree-K Walsh Gram of THIS mask bank. Because Assumption 1 concerns the
    design only (columns depend on masks/basis, not on rho or g_rho), this is a
    pure, model-free, reference-free quantity computable in milliseconds from the
    same X the OLS fit uses. The Tier-1b transfer study (R1.4) shows Cest does
    NOT collapse in pK/N across (d,K), so no single frozen value is correct; the
    honest floor measures Cest per run. Falls back to CONSTANTS.C_FLOOR only if
    the Gram is too ill-conditioned to invert (the fit itself would then fail).
    """
    X = design_matrix(Z, K)
    Xs, _ = standardize_columns(X)
    N = Xs.shape[0]
    G = (Xs.T @ Xs) / N
    try:
        if np.linalg.cond(G) > 1e10:
            raise np.linalg.LinAlgError
        Ginv = np.linalg.inv(G)
        gamma = float(np.linalg.eigvalsh(G)[0])
        Cinv = op_inf_norm(Ginv)
        gpow = gamma ** (-0.5) if gamma > 0 else float("inf")
        cest = max(gpow, Cinv)
        return cest if math.isfinite(cest) and cest >= 1.0 else CONSTANTS.C_FLOOR
    except np.linalg.LinAlgError:
        return CONSTANTS.C_FLOOR


def floor_from_design(Z: np.ndarray, s_eff: float, K: int,
                      family_wise: bool = True, delta: float = None,
                      split: int = None) -> float:
    """R1.4 (Path A) -- the floor with Cest measured from the realized design Z.

        floor = realized_cest(Z, K) * s_eff * sqrt(2 log(SPLIT*pK/delta) / N)

    This is the honest forward floor: the design constant is read off the same
    mask bank the coefficients are fit on, so there is no transfer claim to
    defend. Use this everywhere a certified decision is made on a real design.
    """
    d = Z.shape[1]
    N = Z.shape[0]
    c = realized_cest(Z, K)
    return floor_value(s_eff, d, N, K, family_wise=family_wise,
                       C_floor=c, delta=delta, split=split)


# =========================================================================== #
#  THE FLOOR  (forward direction) -- one function, used everywhere
# =========================================================================== #
def sigma_eff(sigma_obs: float, m_hat: float, C_m: float = None) -> float:
    """sigma_eff = sigma_obs + C_m sqrt(m): the ONLY data-dependent input to the
    floor. Query noise and mismatch enter through this single scalar."""
    C_m = CONSTANTS.C_M if C_m is None else C_m
    return sigma_obs + C_m * math.sqrt(max(m_hat, 0.0))


# --------------------------------------------------------------------------- #
#  R1.5 -- the FULL Bernstein leakage bound (both terms) and its domination
#          regime. Lemma 1 is Bernstein, not purely sub-Gaussian:
#
#     eta_N  <=  sqrt( 2 m log(.) / N )        (sub-Gaussian term)
#             +  (2/3) * B * log(.) / N        (sub-exponential term)
#
#  with m = m>K,rho the mismatch energy and B = ||r>K,rho||_inf its sup-norm.
#  The sub-exponential term is dominated once
#
#     N  >~  (B^2 / m) * log(.)                (domination regime),
#
#  which holds throughout the feasibility regime N >~ pK; it is then absorbed
#  into the calibrated C_M. We compute both terms so Tier 1 can VERIFY the
#  domination rather than assert it, and so the leakage bound is honest at any N.
# --------------------------------------------------------------------------- #
def leakage_bound_terms(m: float, B: float, d: int, N: int, K: int = 1,
                        delta: float = None, split: int = None):
    """Return (sub_gaussian_term, sub_exponential_term, log_factor) of the
    Bernstein leakage bound. `B` is the sup-norm ||r>K,rho||_inf of the mismatch
    residual (bounded response => finite). The reported eta bound is their sum.
    """
    L = log_pk_over_delta(d, K, delta, split)
    m = max(m, 0.0)
    sub_g = math.sqrt(2.0 * m * L / N)
    sub_e = (2.0 / 3.0) * max(B, 0.0) * L / N
    return sub_g, sub_e, L


def leakage_domination_N(m: float, B: float, d: int, K: int = 1,
                         delta: float = None, split: int = None) -> float:
    """Smallest N above which the sub-exponential term is <= the sub-Gaussian
    term, i.e. N >= (2 B^2 / (9 m)) * log(.). Below this N the Bernstein tail is
    NOT dominated and C_M cannot absorb it; Tier 1 checks its cells clear this.
    """
    m = max(m, 1e-12)
    L = log_pk_over_delta(d, K, delta, split)
    return (2.0 * B ** 2 / (9.0 * m)) * L


def floor_value(s_eff: float, d: int, N: int, K: int = 1,
                family_wise: bool = True, C_floor: float = None,
                delta: float = None, split: int = None) -> float:
    """floor(N, rho) = C_floor * sigma_eff * sqrt(2 log(SPLIT*pK/delta) / N)
    (family-wise), or the single pre-registered-coordinate variant with
    z_{1-alpha}.

    R1.2: the log factor is now the union-bounded log_pk_over_delta(), so the
    floor honestly reflects that the proof spends delta over several events. With
    delta = 1/pK and split = 1 this reduces to the pre-revision sqrt(2 log pK/N).
    """
    C_floor = CONSTANTS.C_FLOOR if C_floor is None else C_floor
    if family_wise:
        L = log_pk_over_delta(d, K, delta, split)
        return C_floor * s_eff * math.sqrt(2.0 * L / N)
    return C_floor * s_eff * CONSTANTS.Z_ALPHA / math.sqrt(N)


def certified_set(beta: np.ndarray, fl: float):
    """Forward rule: certify coordinate S iff |beta_hat_S| > floor. Returns
    (index set, sign vector)."""
    idx = np.where(np.abs(beta) > fl)[0]
    return set(idx.tolist()), np.sign(beta)


# =========================================================================== #
#  THE BUDGET RULE  (backward direction) -- Eq. 8, with the budget constant
# =========================================================================== #
def predict_budget(s_eff: float, beta_min: float, d: int, K: int = 1,
                   family_wise: bool = True, C_budget: float = None,
                   delta: float = None, split: int = None) -> int:
    """Backward: smallest N that pushes the floor below beta_min, family-wise.

        N_pred = ceil( C_budget^2 * sigma_eff^2 * 2 log(SPLIT*pK/delta)
                       / beta_min^2 )

    C_budget (NOT C_floor) is the back-solved signed-detection constant. Using
    C_floor here under-predicts below the feasibility floor ~pK.

    R1.2: the log factor matches the floor's union-bounded log_pk_over_delta(),
    so the backward inversion of the floor stays exact under the split delta.
    """
    C_budget = CONSTANTS.C_BUDGET if C_budget is None else C_budget
    norm = (2.0 * log_pk_over_delta(d, K, delta, split) if family_wise
            else CONSTANTS.Z_ALPHA ** 2)
    return int(math.ceil(C_budget ** 2 * s_eff ** 2 * norm / beta_min ** 2))


def feasibility_floor(d: int, K: int = 1, c: float = 3.0) -> int:
    """Design-conditioning floor ~ c * pK below which the dense fit is singular
    (Reading 2). A target below this is run feasibility-CLAMPED, not infeasible."""
    return int(math.ceil(c * p_K(d, K)))


@dataclass
class BudgetPlan:
    """The backward-direction result for one item/cell."""
    N_pred: int
    N_run: int
    realized_floor: float
    ratio: float            # realized_floor / beta_min  (target <= 1)
    clamped: bool           # ran at feasibility floor, not the resolution budget


def plan_budget(s_eff: float, beta_min: float, d: int, K: int = 1,
                family_wise: bool = True, rng: np.random.Generator = None
                ) -> BudgetPlan:
    """Full backward plan: predict N, clamp to feasibility, report realized
    floor and the conservative ratio.

    R1.4 (Path A): the realized floor at N_run now measures Cest from a sampled
    design of that size (floor_from_design) rather than assuming C_FLOOR=1, so
    the reported realized_floor/beta_min ratio reflects the true design constant.
    A fresh rng is drawn if none is supplied; the design is model-free so this
    adds no query cost.
    """
    N_pred = predict_budget(s_eff, beta_min, d, K, family_wise)
    feas = feasibility_floor(d, K)
    N_run = max(N_pred, feas)
    rng = np.random.default_rng() if rng is None else rng
    Zdesign = sample_masks(N_run, d, rng)
    fl = floor_from_design(Zdesign, s_eff, K, family_wise)
    return BudgetPlan(N_pred=N_pred, N_run=N_run, realized_floor=fl,
                      ratio=fl / beta_min, clamped=(N_pred < feas))


# =========================================================================== #
#  PILOT estimation of the only data-dependent input (sigma_eff)
# =========================================================================== #
@dataclass
class MismatchEstimate:
    """R1.6 -- the held-out mismatch estimate with its provenance.

    m_raw     : the raw difference mse - sigma_obs^2 (CAN be negative).
    m_hat     : the clipped value max(m_raw, 0) fed to the floor.
    negative  : whether clipping was active (m_raw < 0).
    resid_var : the held-out residual variance mse.
    B_hat     : plug-in estimate of ||r>K,rho||_inf (sup residual magnitude),
                used by the R1.5 Bernstein-domination check.
    Clipping at zero can only RAISE m_hat relative to m_raw, i.e. it moves
    sigma_eff in the conservative (upper-bounding) direction; the raw value is
    kept only so callers can log the negative-clip fraction, never fed forward.
    """
    m_raw: float
    m_hat: float
    negative: bool
    resid_var: float
    B_hat: float


def _held_out_residual(Z, y, K, cross_fit):
    """Return the held-out residual vector used by the mismatch estimate."""
    if cross_fit:
        n = Z.shape[0]
        half = n // 2
        resid = np.empty(n)
        for tr, te in [(slice(0, half), slice(half, n)),
                       (slice(half, n), slice(0, half))]:
            beta, b0, _ = ols_fit(Z[tr], y[tr], K)
            yhat = b0 + design_matrix(Z[te], K) @ beta
            resid[te] = y[te] - yhat
        return resid
    beta, b0, _ = ols_fit(Z, y, K)
    yhat = b0 + design_matrix(Z, K) @ beta
    return y - yhat


def estimate_mismatch_detail(Z: np.ndarray, y: np.ndarray, K: int,
                             sigma_obs: float,
                             cross_fit: bool = True) -> MismatchEstimate:
    """Full R1.6 breakdown: raw (possibly negative) and clipped mismatch energy,
    plus a plug-in sup-norm B_hat for the Bernstein term. See MismatchEstimate.
    """
    resid = _held_out_residual(Z, y, K, cross_fit)
    mse = float((resid ** 2).mean())
    m_raw = mse - sigma_obs ** 2
    m_hat = max(m_raw, 0.0)
    # B_hat: sup |r>K|. The held-out residual mixes mismatch and query noise; we
    # subtract the query-noise scale in quadrature-free form only for the
    # ENERGY, and report the raw sup for the (conservative) Bernstein constant.
    B_hat = float(np.max(np.abs(resid))) if resid.size else 0.0
    return MismatchEstimate(m_raw=m_raw, m_hat=m_hat, negative=(m_raw < 0.0),
                            resid_var=mse, B_hat=B_hat)


def estimate_mismatch_from_residual(Z: np.ndarray, y: np.ndarray, K: int,
                                    sigma_obs: float,
                                    cross_fit: bool = True) -> float:
    """m_hat>K = held-out residual variance minus sigma_obs^2 (Appendix C),
    clipped at zero (R1.6). Thin wrapper over estimate_mismatch_detail kept for
    back-compat: existing callers still get the single clipped scalar.

    Upper-biased (conservative): the held-out residual contains both genuine
    mismatch and pilot estimation error. cross_fit removes the in-sample
    pK/N inflation -- the recommended default at K=2. When the raw difference is
    negative (no detectable mismatch beyond query noise), clipping returns 0 and
    sigma_eff collapses to sigma_obs, the correct mismatch-free floor.
    """
    return estimate_mismatch_detail(Z, y, K, sigma_obs, cross_fit).m_hat


def pilot_N0(d: int, K: int = 1) -> int:
    """Cross-fitted pilot size N0 = max(500, 6 pK)."""
    return max(500, 6 * p_K(d, K))


# =========================================================================== #
#  FORWARD evidence containers + collapse-curve machinery (Tier 1 / exact-beta)
# =========================================================================== #
@dataclass
class CollapsePoint:
    """One (x, signed-detection-rate) sample on the shared collapse curve."""
    x: float            # |beta| / floor
    sdr: float          # signed-detection rate at this x


def collapse_curve(x_grid, sdr_values):
    """Bundle an SDR(x) curve and return its 50%-crossing by linear interp.
    The crossing x_0.5 is a single labeled POINT on the shared curve, reported
    only alongside its cross-regime spread -- it is not a standalone constant.
    Returns (list[CollapsePoint], x_0.5 or nan)."""
    pts = [CollapsePoint(float(x), float(s)) for x, s in zip(x_grid, sdr_values)]
    x_half = float("nan")
    for k in range(1, len(sdr_values)):
        if sdr_values[k - 1] < 0.5 <= sdr_values[k]:
            lo, hi = x_grid[k - 1], x_grid[k]
            plo, phi = sdr_values[k - 1], sdr_values[k]
            x_half = lo + (0.5 - plo) * (hi - lo) / (phi - plo)
            break
    return pts, x_half


def cov(values) -> float:
    """Coefficient of variation std/mean -- the collapse-tightness statistic.
    Low CoV across regimes IS the claim; the mean value is secondary."""
    a = np.asarray([v for v in values if v is not None and not math.isnan(v)],
                   dtype=float)
    if a.size < 2 or a.mean() == 0:
        return float("nan")
    return float(a.std() / a.mean())


def false_sign_rate(beta_run: np.ndarray, beta_exact: np.ndarray, fl: float,
                    fl_exact: float):
    """Direct forward test against a known/exact beta. A coordinate is scored
    only if BOTH the run and the exact fit resolve it (an exact-unresolved coord
    carries no trustworthy ground-truth sign). Returns (n_false, n_scored)."""
    run_cert = np.abs(beta_run) > fl
    exact_cert = np.abs(beta_exact) > fl_exact
    scored = run_cert & exact_cert
    false = scored & (np.sign(beta_run) != np.sign(beta_exact))
    return int(false.sum()), int(scored.sum())


# =========================================================================== #
#  WORKFLOW DIAGNOSTICS (kept SEPARATE from the guarantee, by design)
# =========================================================================== #
@dataclass
class DiagnosticTrace:
    """Secondary, prefix-construction-coupled diagnostics. Reported in the
    appendix, explicitly labeled as near-guaranteed by nested prefixes -- NOT
    theorem evidence. The guarantee proper is sign_flips (forward)."""
    count_monotone: bool = True
    set_nested: bool = True
    sign_flips: int = 0          # THE guarantee: certified coords reversing sign
    n_compared: int = 0          # denominator: certified-in-both-budgets checks
    floor_first: float = None
    floor_last: float = None
    cert_first: int = None
    cert_last: int = None
    floor_first_old: float = None   # R1.4: old frozen-C_FLOOR floor (first rung)
    floor_last_old: float = None    # R1.4: old frozen-C_FLOOR floor (last rung)
    cest_first: float = None        # R1.4: realized Cest at first rung
    cest_last: float = None         # R1.4: realized Cest at last rung


def sweep_prefix_ladder(Zbank: np.ndarray, ybank: np.ndarray, N_list,
                        s_eff: float, d: int, K: int,
                        family_wise: bool = True) -> DiagnosticTrace:
    """Apply the single-budget theorem at each rung of a prefix-nested ladder.

    R1.4 (Path A): the floor at each rung uses Cest MEASURED from that rung's
    realized design prefix Zbank[:N] (floor_from_design), not the frozen C_FLOOR.
    We also record the old frozen-C_FLOOR floor at the first/last rung so drivers
    can report the before/after floor inflation.

    Primary output: sign_flips (the guarantee -- must be 0). Secondary outputs:
    count_monotone / set_nested (workflow diagnostics, near-guaranteed by the
    prefix construction and therefore reported as such, not as evidence).
    """
    tr = DiagnosticTrace()
    prev_set = prev_beta = None
    prev_count = -1
    for N in N_list:
        Zc = Zbank[:N]
        beta, _, _ = ols_fit(Zc, ybank[:N], K)
        fl = floor_from_design(Zc, s_eff, K, family_wise)       # realized Cest
        cur_set, _ = certified_set(beta, fl)
        if tr.floor_first is None:
            tr.floor_first, tr.cert_first = fl, len(cur_set)
            tr.floor_first_old = floor_value(s_eff, d, N, K, family_wise)
            tr.cest_first = realized_cest(Zc, K)
        tr.floor_last, tr.cert_last = fl, len(cur_set)
        tr.floor_last_old = floor_value(s_eff, d, N, K, family_wise)
        tr.cest_last = realized_cest(Zc, K)
        if len(cur_set) < prev_count:
            tr.count_monotone = False
        if prev_set is not None and len(prev_set - cur_set) > 1:
            tr.set_nested = False
        if prev_beta is not None:
            inter = list(prev_set & cur_set)
            tr.n_compared += len(inter)
            tr.sign_flips += sum(np.sign(beta[i]) != np.sign(prev_beta[i])
                                 for i in inter) if inter else 0
        prev_set, prev_beta, prev_count = cur_set, beta, len(cur_set)
    return tr