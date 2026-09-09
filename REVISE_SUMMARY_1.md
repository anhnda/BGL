# BudgetLIME — Revision Summary

*Finite-Budget Certification for LIME-Style Local Surrogates* — code revision log
for the reviewer response. This file records **what changed in the code**, **which
reviewer points are resolved**, and **what still has to be done** (re-runs and
manuscript edits). It is a companion to `README.md`, not a replacement.

---

## 1. TL;DR

The revision implements every **correctness** fix raised by the reviewers at the
single numerical source of truth (`bl_core.py`), so all tiers inherit them, and
adds one new **design study** (`tier1b_cest_transfer.py`) that settles the
reviewer's most serious point. The floor's forward constant is no longer frozen:
it is measured per run from the realized design.

| Item | Reviewer | Status (code) | Status (paper) |
|------|----------|---------------|----------------|
| δ-budget split | R1.2 | ✅ done | ✍️ theorem statement edit owed |
| Bernstein sub-exp term | R1.5 | ✅ done | ✍️ Lemma 1 proof edit owed |
| Negative-mismatch / clipping | R1.6 | ✅ done (diagnostic) | ✍️ Appendix C edit owed |
| Leakage-constant consistency | R1.3 | ✅ resolved (one normalizer) | ✍️ Table 2 + one sentence |
| Cest transfer | R1.4 | ✅ dissolved (per-run Cest) | ✍️ Eq. 6 reframe + study figure |
| UCB pilot for σ_eff | R2.1 | ⏳ not started | — |
| Ridge ℓ∞ adaptation | R2.2 | ⏳ not started | — |
| Scope / "LIME-style" caveat | R1.1, R3 | ⏳ not started | ✍️ intro/title qualification |
| Item-level (per-run) failure rate | R1.7 | ⏳ not started | — |
| Enumeration power / miss-rate | R1.9 | ⏳ not started | — |
| Residual-based Wald comparator | R1.10 | ⏳ not started | — |
| Security/LLM related work + cite | R1.11 | ⏳ not started | ✍️ related work |

Legend: ✅ done · ⏳ open · ✍️ manuscript text still owed even where code is done.

**Frozen constants after this revision** (`bl_core.CONSTANTS`):
`C_M = 0.833`, `C_FLOOR = 1.0` (ideal/fallback only), `C_BUDGET = 1.535`,
`DELTA_SPLIT = 3`.

---

## 2. What changed, by reviewer point

### R1.2 — δ-budget accounting  ✅ (code)

**Problem.** Theorem 1 was stated at `1−δ` while its proof spends several `1−δ`
events (design conditioning, the sub-Gaussian query-noise maximum, the Lemma 1
mismatch bound).

**Change.** Added `DELTA_SPLIT = 3` and a single helper
`bl.log_pk_over_delta(d, K, δ, split)` returning `log(split·pK/δ)`. Routed the
floor (`floor_value`) and the budget rule (`predict_budget`) through it. With
`δ = 1/pK` this turns the old `2 log pK` into `2 log pK + 2 log 3`, a bounded
additive correction (floor inflation ≈ 6.8 % at `pK = 50`, ≈ 3.8 % at
`pK ≈ 1225`). The pilot scale is handled as Theorem 1's explicit conditioning
hypothesis (hence `split = 3`, not 4). `split=1, δ=1/pK` reproduces the old
factor exactly for audits.

**Files.** `bl_core.py` (helper, `floor_value`, `predict_budget`),
`tier1_synthetic.py` (backward calibration uses the same factor).

### R1.5 — sub-exponential Bernstein term in Lemma 1  ✅ (code)

**Problem.** The leakage bound was passed off as purely sub-Gaussian; the
Bernstein sub-exponential term was dropped rather than carried and dominated.

**Change.** Added `bl.leakage_bound_terms` (both terms) and
`bl.leakage_domination_N` (the `N ≳ (2B²/9m)·log(.)` threshold). Tier 1's
`leakage` calibration now prints the sub-exp term, `N_dom`, and a per-cell
`dom?` flag, and **verifies** domination (25/25 cells pass in the run on file).
Tier 2 reports the fraction of items inside the dominated regime (`dom%`).

**Files.** `bl_core.py`, `tier1_synthetic.py`, `tier2_blackbox.py`.

### R1.6 — negative mismatch estimate + clipping bias  ✅ (code, diagnostic)

**Problem.** The held-out mismatch estimate is a difference of variances and can
be negative; the treatment and clipping bias were unstated.

**Change.** `bl.estimate_mismatch_detail` returns the **raw** (possibly negative)
value, the clipped value fed to the floor, a `negative` flag, and a plug-in
`B_hat` for the R1.5 check. Clipping at zero only raises `σ_eff` (conservative).
Tier 2 logs the per-cell fraction of pilot draws with `m_hat < 0` (`neg%`),
expected nonzero mainly in near-degenerate / deterministic cells — the
conditional-clause mechanism (ViSoBERT/zero).

**Not covered here (by design):** the *substantive* repair — an over-estimated
`σ_obs` hiding real mismatch — needs the **R2.1** upper-confidence estimator,
which is not started. R1.6 here makes the failure **visible**, not **fixed**.

**Files.** `bl_core.py`, `tier2_blackbox.py`.

### R1.3 — leakage-constant consistency  ✅ (resolved)

**Problem.** `C_m` was calibrated against a different normalizer than the floor
carried, so consistency could not be read off.

**Change.** One canonical log factor (`log_pk_over_delta`) is used across Lemma 1,
the floor, and the `C_m` calibration. Consequences, both benign and stated in the
`Constants` docstring:
- the larger normalizer rescales the raw ratio to ≈ 0.814 (its product
  `C_m·√normalizer` matches the old 1.24 to 0.01 % — a pure re-expression);
- `C_m` is an asymptotic sqrt-regime quantity, so it is **frozen from the
  large-N rows** (`N ≥ 2000`), lifting it to **0.833**; the ≈ 2.4 % difference is
  exactly the R1.5 sub-exp inflation the large-N freeze removes, quantified per
  cell in the `subexp` column. All-N mean (0.81) is printed for transparency.

**Files.** `bl_core.py` (`Constants` docstring, frozen value),
`tier1_synthetic.py` (two-line all-N vs large-N report).

### R1.4 — Cest transfer  ✅ (dissolved, Path A)  ← the reviewer's "most serious"

**Problem.** `Cest = max{γ^{-1/2}, ‖Σ̂⁻¹‖∞}` (the forward-floor constant, Eq. 6)
is a max absolute row sum that grows with `pK`, yet was calibrated at
`(d=30, K=1)` and frozen for `d=49` and `K=2`.

**New study.** `tier1b_cest_transfer.py` (pure numpy, no model):
- **Q1** — `Cest` grows with `pK` at fixed `N` (premise confirmed).
- **Q2** — `Cest` does **not** collapse in `pK/N`: at fixed ratio it splits by
  `pK`/(d,K) with 40–58 % within-bin spread. **No single frozen scalar is
  correct.**
- **Q3** — empirical `Cest` at the deployed points is **1.5–3.8×**, not the
  frozen `C_FLOOR = 1.0`; a floor computed at 1.0 is anti-conservative by those
  factors.

**Resolution (Path A).** `Cest` is now **measured per run** from the realized
design via `bl.realized_cest(Z, K)` / `bl.floor_from_design(Z, s_eff, K)` — a
model-free, reference-free quantity read off the same Gram the OLS fit uses.
Every certified decision uses it: the Tier-2 forward ladder
(`sweep_prefix_ladder`), the exact-β check (`exact_sign_check`), the backward
plan (`plan_budget`), the reseed audit (`tier2b`), and the baseline comparison.
`C_FLOOR` is retained only as the orthonormal ideal / ill-conditioned fallback.
There is **no transfer claim left to defend** — nothing is transferred.

The Tier-2 report prints per cell the realized `Cest` and the floor-inflation
factor vs the old frozen floor, so the before/after is visible.

**Files.** `bl_core.py` (`measure_conditioning`, `op_inf_norm`,
`ConditioningProbe`, `realized_cest`, `floor_from_design`, `plan_budget`,
`sweep_prefix_ladder`, `DiagnosticTrace` fields), `tier1b_cest_transfer.py`
(new), `tier2_blackbox.py`, `tier2b_reseed.py`, `baselines.py`.

---

## 3. Files touched

| File | Status | Change |
|------|--------|--------|
| `bl_core.py` | modified | δ-split helper; Bernstein terms; mismatch detail; conditioning probe; **per-run realized floor**; constants re-frozen |
| `tier1_synthetic.py` | modified | leakage calibration uses split normalizer + Bernstein/domination columns; large-N `C_m` freeze; backward `C_budget` under split factor |
| `tier1b_cest_transfer.py` | **new** | R1.4 Cest-transfer study (Q1 mechanism / Q2 collapse / Q3 operating point) |
| `tier2_blackbox.py` | modified | realized floor in forward ladder + exact-β; `Cest`/`flr×`/`neg%`/`dom%` report columns |
| `tier2b_reseed.py` | modified | realized floor in `_fit_certify` |
| `baselines.py` | modified | floor rule uses realized `Cest` on the shared bank |
| `tier3_feasibility.py` | unchanged | (uses `predict_budget`, already routed through the split factor) |
| `bl_models.py` | unchanged | the only torch file; no math changes |
| `README.md` | modified | Correctness-fixes section; R1.4 Path A section; Tier-1b row; revised Table 2 |

---

## 4. What YOU must do next

### 4.1 Re-run (the constants and the floor both changed)

Run order — **all forward-tier numbers must be regenerated**, because the floor
is now 1.5–3.8× larger and the constants moved:

```bash
# 1. Confirm the frozen constants print what is frozen.
python tier1_synthetic.py all          # expect C_m=0.83, C_budget=1.54

# 2. The new R1.4 study (already run once; re-run if seeds/grid change).
python tier1b_cest_transfer.py all

# 3. Forward-tier guarantees UNDER THE REALIZED FLOOR (the important re-run).
python tier2_blackbox.py image --beta_min 0.05 --N_ladder 512,1000,2000,4000 \
    --images_dir image_samples --glob "*.JPEG"
python tier2_blackbox.py nlp --K 1 --references mask,pad,zero \
    --backbones distilbert,roberta,visobert --beta_min 0.02 \
    --N_ladder 512,1000,2000,4000 --sentences text_samples/sst2_short.txt
python tier2_blackbox.py exact-nlp --K 2 --subset 10 --max_d 13 \
    --sentences text_samples/sst2_short.txt
python tier2b_reseed.py nlp --backbones visobert --references zero --N 2000 
python tier2b_reseed.py image --backbones resnet50 --references mean --N 2000 
python tier3_feasibility.py
python baselines.py image --backbones resnet50 --references mean --N 2000 --B 200
```

**What to watch on re-run:**
- **`flr×` column** — how much each cell's floor grew vs the old frozen floor.
- **Certified sets shrink**, most at low budgets / high `pK/N`. This is the honest
  resolution, not a regression.
- **Re-verify the guarantees.** Zero sign-flips (forward) and zero false-signs
  (exact-β, K=2) were generated under a floor 1.5–3.8× too small. They most
  likely survive — a larger floor certifies fewer but stronger coordinates — but
  this must be **shown**, not assumed. If they survive, the correctness story is
  stronger, because it holds under the honest floor.
- **Backward ratios** move (`plan_budget` now measures `Cest` at `N_run`).

### 4.2 Reproducibility caveat

`plan_budget` samples a fresh design to measure the realized floor, so the
backward ratio is mildly stochastic run-to-run. Pass a seeded `rng` if you want
the backward table reproducible (signature already accepts one; drivers don't yet
thread a seed in).

### 4.3 Manuscript edits still owed (code done, text not)

1. **Theorem 1** — write the `δ₁+δ₂+δ₃=δ` split (the actual R1.2 discharge).
2. **Lemma 1 proof** — carry the Bernstein sub-exp term, cite the domination
   table (R1.5).
3. **Table 2 + one sentence** — new constants (`C_m=0.833`, `C_budget=1.535`);
   state the R1.3 normalizer choice and the large-N freeze.
4. **Eq. 6 reframe (R1.4)** — the forward constant is measured per run, not
   frozen; use the Tier-1b Q2 non-collapse as the justification figure.
5. **Appendix C** — the clipping rule, its conservative direction, and `neg%`
   reporting (R1.6).

### 4.4 Still open (new work, by design not in this pass)

- **R2.1** — upper-confidence σ_eff estimator (the real repair for R1.6 /
  ViSoBERT-zero; converts the conditional guarantee into an unconditional one).
- **R2.2** — Ridge-penalized ℓ∞ adaptation (dedicated section).
- **R1.1 / R3** — "LIME-style" scope caveat (uniform masks, no locality kernel →
  Walsh/Banzhaf object); title/intro qualification and the degree-one ↔ Banzhaf
  remark.
- **R1.7** — per-run item-level failure rate in the reseed audit (the faithful
  test of the simultaneous probability statement).
- **R1.9** — enumeration power / miss-rate (how many true non-zeros failed to
  clear the floor).
- **R1.10** — residual-based Wald comparator (fair competitor) + bootstrap note.
- **R1.11** — security/LLM related-work paragraph and citation.

---

## 5. One-line status

**Correctness fixes (R1.2, R1.5, R1.6) implemented; R1.3 resolved; R1.4 dissolved
by measuring the floor constant per run.** Forward-tier numbers must be
regenerated under the realized floor, and the matching manuscript edits written,
before the response is complete. R2.1/R2.2 and the empirical additions
(R1.7/R1.9/R1.10) remain open.