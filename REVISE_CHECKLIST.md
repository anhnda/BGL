# BudgetLIME — Reviewer Points Checklist

Point-by-point action list for the revision of *Finite-Budget Certification for
LIME-Style Local Surrogates*. Each item: **status**, **what to do**, **where**,
and **done-when** (acceptance criterion). Companion to `REVISE_SUMMARY.md`.

Status legend: `[x]` done · `[~]` partially done (code done, text owed) · `[ ]` not started.

Editor's overall instruction: report results accurately, rewrite any overstated
conclusions, and fully explain the limitations.

---

## Reviewer 1 (revise)

- [~] **R1.1 — "LIME-style" scope caveat.** Add an early qualification that with a
  uniform product measure and no locality kernel, the estimand is a
  Walsh–Fourier / Banzhaf-type object, not kernel-weighted LIME. Add the explicit
  degree-one ↔ Banzhaf-value remark.
  *Where:* intro/§2 (and a title qualification).
  *Done when:* a reader cannot mistake the proved scope for canonical LIME; Banzhaf
  remark present.

- [x] **R1.2 — δ-budget accounting.** Split δ across the failure events (design
  conditioning + query-noise maximum + mismatch bound); state the theorem at the
  union-bounded level.
  *Where (code):* `bl_core.log_pk_over_delta`, `floor_value`, `predict_budget`,
  `DELTA_SPLIT=3`. *Where (paper):* Theorem 1 statement + Appendix A union-bound
  sentence.
  *Done when:* theorem reads `δ₁+δ₂+δ₃=δ` (or `1−cδ`); floor carries
  `log(3·pK/δ)`. **Code done; theorem text owed.**

- [~] **R1.3 — leakage-constant consistency.** Make Lemma 1's normalizer, the
  floor, and the `C_m` calibration use ONE canonical log factor; report the new
  constant and explain the value change.
  *Where (code):* `bl_core` (frozen `C_M=0.833`), `tier1_synthetic.calibrate_leakage`.
  *Where (paper):* Table 2 + one sentence.
  *Done when:* Table 2 shows `C_m=0.833` with the normalizer + large-N-freeze
  sentence; product-invariance noted. **Code done; Table 2 sentence owed.**

- [x] **R1.4 — Cest transfer (MOST SERIOUS).** Provide evidence Cest is/ isn't
  stable in `pK/N`; fix accordingly.
  *Where (code):* `tier1b_cest_transfer.py` (study), `bl_core.realized_cest` /
  `floor_from_design` (per-run measurement wired into all tiers).
  *Resolution:* study shows NO collapse (40–58% split) → Cest measured per run,
  not frozen. *Where (paper):* reframe Eq. 6's constant as per-run-measured; use
  Tier-1b Q2 as the figure.
  *Done when:* forward-tier results regenerated under the realized floor and the
  Eq. 6 reframe written. **Code done; re-run + paper reframe owed.**

- [~] **R1.5 — carry the sub-exponential term.** Carry Bernstein's
  `(2/3)B log(.)/N` term, then show it is dominated for `N ≳ (B²/m)log(.)`.
  *Where (code):* `bl_core.leakage_bound_terms`, `leakage_domination_N`; Tier 1
  `dom?` column (25/25 pass). *Where (paper):* Lemma 1 proof.
  *Done when:* proof carries the term and cites the domination regime.
  **Code done; proof text owed.**

- [~] **R1.6 — negative mismatch estimate + clipping bias.** State the clipping
  rule, its (conservative) bias direction, and the negative case.
  *Where (code):* `bl_core.estimate_mismatch_detail` (raw/clipped/flag/B_hat);
  Tier 2 `neg%` column. *Where (paper):* Appendix C.
  *Done when:* Appendix C states the rule + reports `neg%`. NOTE: the substantive
  repair is R2.1. **Code done (diagnostic); Appendix C text owed.**

- [ ] **R1.7 — per-run (item-level) failure rate.** The theorem's failure event
  is per-run and simultaneous; report the fraction of draws in which ANY certified
  coordinate disagrees (not the per-coordinate rate), especially for the
  weak-signal ViSoBERT/zero cell.
  *Where:* `tier2b_reseed.py` — add an item-level violation statistic alongside
  the existing per-coordinate one.
  *Done when:* reseed table shows per-run failure rate vs δ target.

- [ ] **R1.8 — stability vs correctness wording.** Make explicit that cross-seed
  agreement is *stability*, not *correctness*; only enumeration tests correctness.
  Sweep §5.2 "takeaway" and §5.4 for blurred usage.
  *Where:* paper text only.
  *Done when:* no passage conflates the two.

- [ ] **R1.9 — enumeration power / miss-rate.** Report how many genuinely non-zero
  population coefficients FAILED to clear the floor (a certificate that resolves
  little is trivially correct).
  *Where:* `tier2_blackbox.exact_sign_check` / `report_exact` — add
  false-negative (miss) count against exact β.
  *Done when:* enumeration result reports power alongside zero false-signs.

- [ ] **R1.10 — fair Wald comparator + bootstrap note.** Replace the query-noise-
  only Wald with a Wald whose variance is estimated from FITTED RESIDUALS (absorb
  mismatch); add a note on why the bootstrap on the same mask bank also cannot see
  deterministic mismatch.
  *Where:* `baselines.certify_wald` — add a residual-variance variant; Appendix D
  text.
  *Done when:* Appendix D compares against the fair Wald and explains the
  bootstrap limitation.

- [ ] **R1.11 — security/LLM related work + citation.** Add a paragraph
  contrasting certified surrogate coefficients with *generated* explanations
  (no detection floor); cite Rizzo et al., *Advanced LLM Prompting Strategies for
  Reentrancy Classification and Explanation in Smart Contracts*, BlockTEA 2025,
  37–56.
  *Where:* related work.
  *Done when:* paragraph present; citation added (only if judged genuinely
  relevant — defensible as a generated-vs-certified contrast).

---

## Reviewer 2 (revise)

- [ ] **R2.1 — conservative upper-confidence pilot for σ_eff.** Define and
  implement a UCB-style estimator that upper-bounds σ_eff with high probability
  (add a one-sided margin to Appendix-C's estimate), folding its failure
  probability into the δ budget. This is the real repair for R1.6 and the
  ViSoBERT/zero anti-conservative case; it removes the "conditional on σ_eff"
  caveat.
  *Where:* `bl_core` (new estimator) + `pilot_sigma_eff`; paper Appendix C /
  a new subsection.
  *Done when:* the guarantee is stated unconditionally (σ_eff valid w.h.p.), at
  least at K=1; ViSoBERT/zero no longer relies on the conditional clause.

- [ ] **R2.2 — Ridge-penalized ℓ∞ adaptation.** Dedicated section deriving the
  floor under ridge: shrinkage bias term, modified `(Σ̂+λI)⁻¹` normalizer,
  resulting bias–variance floor.
  *Where:* new theory subsection (+ optional `bl_core` ridge fit).
  *Done when:* ridge floor stated with the bias term explicit (theorem or rigorous
  sketch).

---

## Reviewer 3 (accept)

- [ ] **R3.1 — scope honesty (overlaps R1.1).** Tighten abstract/intro/title so
  the advertised "LIME-style" scope matches what is proved; move the
  locality-kernel / feature-selection / Ridge exclusions earlier than the
  Limitations section and note that excluding locality weighting weakens the
  connection to canonical LIME.
  *Where:* abstract, intro, title.
  *Done when:* scope caveat appears early, not only in Limitations.

---

## Cross-cutting re-runs (blocked by code changes above)

- [ ] **Regenerate all forward-tier numbers under the realized floor (R1.4).**
  Tier 2 image + nlp, exact-nlp, Tier 2b, Tier 3, baselines. Floor is now
  1.5–3.8× larger.
  *Done when:* zero-sign-flip (forward) and zero-false-sign (exact-β, K=2) are
  RE-CONFIRMED under the realized floor, or the change is reported honestly.

- [ ] **Update Table 2 / Table 3 / Table 4 numbers** to the re-run values
  (constants: `C_m=0.833`, `C_budget=1.535`; realized `Cest` per cell).

---

## Suggested execution order

1. **Re-runs** (unblock the tables): forward-tier under the realized floor.
2. **Manuscript correctness edits** already backed by code: R1.2 theorem, R1.5
   proof, R1.3 Table 2 sentence, R1.4 Eq. 6 reframe, R1.6 Appendix C.
3. **New empirical items**: R1.7, R1.9, R1.10 (share infra with existing code).
4. **New theory**: R2.1 (UCB σ_eff) then R2.2 (Ridge).
5. **Framing / related work**: R1.1, R1.8, R1.11, R3.1.

---

## Quick status count

- Done (code): R1.2, R1.3, R1.4, R1.5, R1.6 core mechanisms.
- Text owed on otherwise-done items: R1.2, R1.3, R1.4, R1.5, R1.6.
- Not started: R1.1, R1.7, R1.8, R1.9, R1.10, R1.11, R2.1, R2.2, R3.1, and the
  forward-tier re-runs.