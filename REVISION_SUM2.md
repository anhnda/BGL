# BudgetLIME Revision — Handoff Summary

Paper: **Finite-Budget Certification for LIME-Style Local Surrogates** (Springer, `sn-jnl` class).
Repo: https://github.com/anhnda/BudgetLIME

This file summarizes the whole revision so a new session can continue. Read it top to bottom.

---

## 0. Files in `/mnt/user-data/outputs/`

| File | What it is | Status |
|---|---|---|
| `main.tex` | Full manuscript, with `\rev{}` toggle for red-marked changes | **latest, authoritative** |
| `bl_core.py` | Core numerics (floor, leakage, reseed, UCB pilot, power/miss) | latest |
| `tier1_synthetic.py` | Tier-1 calibration + `sweep_d_stability` | latest |
| `tier2_blackbox.py` | Tier-2 forward/backward + `exact-nlp` (power/miss) | latest |
| `tier2b_reseed.py` | Tier-2b independent reseeding audit (rewritten from scratch) | latest |
| `baselines.py` | Appendix D comparators (fair Wald, bootstrap) | latest |
| `responses.tex` + `response_reviewer{1,2,3}.tex` + `responses.pdf` | Point-by-point response letter | done, may need number sync |
| `rizzo2025_bibentry.bib` | BibTeX entry for R1.11 to paste into `sn-bibliography.bib` | done |

**IMPORTANT:** always work from `outputs/main.tex`. Earlier uploads (`main_v1.tex`, etc.) are stale.

---

## 1. The `\rev{}` toggle mechanism

Preamble has:
```latex
\newif\ifreview
\reviewfalse   % \reviewtrue -> red-marked version
\newcommand{\rev}[1]{\ifreview{\color{red}#1}\else{#1}\fi}
```
Uses **group-scoped `\color`** (not `\textcolor`) so a `\rev{}` block can span paragraphs/footnotes
without "Paragraph ended before \@textcolor" errors. All revision text is wrapped in `\rev{}`.

**Cannot compile with `sn-jnl.cls` here** (class not in this TeX install). Verified instead by
extracting the body into an `article` wrapper — 0 content errors both `\reviewtrue`/`\reviewfalse`.
User must build with real `sn-jnl.cls` to confirm final page count.

---

## 2. Reviewer points — status

Three reviewers: R1 (revise, 11 points), R2 (revise, 2 points), R3 (accept, 1 point).

| Point | What it asked | Resolution | Code | Done? |
|---|---|---|---|---|
| **R1.1 / R3.1** | "LIME-style" scope: Walsh/Banzhaf, not kernel-weighted | Remark 1 (§3.1) + abstract + intro + title footnote; degree-1 = Banzhaf value | — | ✅ |
| **R1.2** | δ-budget split (theorem states 1−δ but spends it 3×) | Union bound over ν events, floor carries `log(ν·pK/δ)`, ν=3 (4 with UCB) | ✅ | ✅ |
| **R1.3** | Leakage-constant normalizer consistency | One canonical `log(ν·pK/δ)` in Lemma 1, floor, and Cm calibration | ✅ | ✅ |
| **R1.4** | **(most serious)** Cest frozen & transferred but grows with pK | Cest **measured per run**; Tier-1b transfer study; Table 3 has per-run Cest column | ✅ | ✅ |
| **R1.5** | Carry Bernstein sub-exponential term | Lemma 1 shows both terms + domination regime `N ≳ 2B²/9m · L`; `dom%` col | ✅ | ✅ |
| **R1.6** | Negative mismatch estimate / clipping bias | Clip to 0 (conservative, raises σeff); `neg%` col; Appendix C | ✅ | ✅ |
| **R1.7** | Per-run (item-level) failure rate, not per-coordinate | Table 4 has both; visobert/zero per-run 0.21 vs per-coord 0.056 | ✅ | ✅ |
| **R1.8** | Stability ≠ correctness | Reseeding = stability; correctness only in enumeration tier | — | ✅ |
| **R1.9** | Enumeration power/miss | §5.3: guarantee margin 35/35 (power 1.0), resolution recovery 110/336 | ✅ | ✅ |
| **R1.10** | Fair Wald (residual variance) + bootstrap note | Appendix D: fair Wald 0/10 certify-all; floor wins on simultaneity+budget | ✅ | ✅ |
| **R1.11** | Security/LLM related work + cite Rizzo BlockTEA 2025 | Related-work paragraph + `\cite{rizzo2025}` | — | ✅ |
| **R2.1** | Conservative upper-confidence pilot for σeff | `sigma_eff_ucb` (empirical-Bernstein, one-sided); Appendix C.1; `--pilot ucb` | ✅ | ✅ |
| **R2.2** | Ridge ℓ∞ adaptation | Appendix E: floor + shrinkage bias `b_λ`, normalizer `(Σ̂+λI)⁻¹` (theory only) | — | ✅ |

**All 14 points addressed.** Reviewer response letter (`response_reviewer*.tex`) written for each.

---

## 3. Constants — the part that got messy (READ CAREFULLY)

There are **three distinct constants**. Do NOT conflate them (an earlier draft did, causing
self-contradictions the user caught):

- **`Cm`** (leakage, Lemma 1) — normalized, **frozen**. Value = **0.830**.
- **`Cbud`** = `C_{\mathrm{bud}}` (budget-*planning*, Eq. 8) — normalized, **frozen**. Value = **1.552**.
  - Used ONLY to predict N *before* a design exists (can't measure Cest yet).
  - The certificate NEVER uses Cbud; once a run exists it re-measures Cest.
- **`Cest`** (floor, Eq. 6) = `max{γ^{-1/2}, ‖Σ̂⁻¹‖∞}` — raw, **per-run, NOT frozen** (R1.4).

### 3.1 Frozen value = MEAN over d (not a single d!)

Critical fix: freezing `Cm=0.833` was WRONG — that's the d=30 value. Since we *argue* d-stability,
we must freeze the **mean over d ∈ {15,24,30,49}**:

```
 d    pK   Cm      Cbud    Cest_emp
15    16   0.820   1.604   1.412
24    25   0.794   1.561   1.552
30    31   0.833   1.535   1.767
49    50   0.871   1.510   2.213
MEAN       0.830   1.552   (per run)
```

`bl_core.CONSTANTS`: `C_M = 0.830`, `C_BUDGET = 1.552` (updated). `C_FLOOR = 1.0` (ideal only).

### 3.2 CoV — which one, and how computed (user's last question)

CoV = std/mean. Two *populations* to compute over — they turned out ≈ equal:
- **between-d** (std/mean of the 4 per-d means) = **0.039** (Cm), 0.026 (Cbud), 0.202 (Cest)
- **pooled** (all ~40 large-N cells across all d) = **0.038** (Cm) — nearly identical
- **within-d** (per d, over m/N cells) = 0.004–0.031 (small, so between-d hides nothing)

`ddof`: `bl.cov()` uses ddof=0 (→ 0.033); `sweep_d_stability` used ddof=1 (→ 0.039). **Minor
difference.** Currently the manuscript reports **0.039 / 0.026 / 0.202** (between-d, ddof=1).

**OPEN DECISION (user said "not much difference, move on"):** pick ONE convention and make
`bl.cov` and `sweep_d_stability` consistent. Recommend: keep between-d (answers R1.4's "stable
across d?"), note in text it ≈ pooled. If unifying ddof, re-sync the three numbers everywhere
(Table 2, its caption, §5.1 Result, Appendix A proof).

### 3.3 Downstream impact of 0.833→0.830, 1.535→1.552

Tiny: σeff shifts −0.36%, N_pred +2.2%. Changes NO conclusion (0 flips stay 0; Cest is per-run
so Table 3 unaffected). No experiments need re-running for this.

### 3.4 Cest in Table 2 vs Table 3 differ — this is expected

- Table 2 (dsweep) Cest: 1.41/1.55/1.77/2.21 at fixed reference budget `N=max(6pK,2000)` — isolates d.
- Table 3 (tier2) Cest: 1.26 (d15) / 1.40 (d24) / 1.79 (d49) at *actual deployment* budgets.
- Caption of Table 2 now says this explicitly. Both are "per-run Cest" but at different N. **Keep both.**

---

## 4. Page-count reduction (target ≤ 20; last real build was 22)

Done so far (all preserve every `\rev{}` reviewer-answer):
- Removed §5.6 Summary (dup + had stale "256"); removed Table D1 (true-by-construction, folded to prose).
- Merged/compressed intro, Related Work (8→4 paras), §5.1 Metrics, §5.2 Result+Takeaway.
- Tables 1 & 3 → `\footnotesize` + `\arraystretch{0.9}`; Fig 1 → `0.88\linewidth`.
- Global `\parskip = 0pt plus 0.5pt`, tightened display/float spacing.

**Estimated ~21 pages now. Still needs a real `sn-jnl` build to confirm.** If > 20, safe next cuts
(none touch reviewer answers): trim Remark 2 (Lasso, Appendix A); merge Appendix B paras; Table D2 → footnotesize.

---

## 5. Symbol notes (things renamed — keep consistent)

- `SPLIT` → **`\nu`** (ν) everywhere — the union-bound event count (ν=3, or 4 with UCB pilot).
  Was `\mathrm{SPLIT}` (looked like code). Defined in Table 1 notation + Lemma 1.
- `C_est` for the budget constant → **`\Cbud`** = `C_{\mathrm{bud}}` (macro added). Cest now means
  ONLY the per-run floor constant.
- Removed the stray internal tag "(R1.7)" that had leaked into a Table 4 caption.
- Fixed "constants frozen from Table 2" (contradicted R1.4) → "leakage/budget constants frozen;
  forward Cest per run" in: tier overview, §5.2 title, Table 3 caption, Result-takeaway, Table 2 caption.

---

## 6. What still needs doing (for the next session)

1. **Build with real `sn-jnl.cls`**, confirm ≤ 20 pages; if not, apply §4 safe cuts.
2. **CoV convention** (§3.2): unify ddof between `bl.cov` and `sweep_d_stability`, re-sync the 3 numbers
   if changed. (Low priority — user said difference is negligible.)
3. **Paste `rizzo2025` entry** into `sn-bibliography.bib`.
4. **Appendix numbering**: verify Ridge = App. E, UCB = App. C.1 match cross-refs; check if
   `compare.tex` (baseline appendix) is `\input` anywhere — if so, reconcile with inline App. D.
5. **Sync response letter numbers** (`response_reviewer*.tex`) with final manuscript values
   (esp. Cm=0.830, Cbud=1.552, CoV figures, per-run Cest column).
6. **Re-run confirmation commands** (all reproduce current numbers):
   ```bash
   python tier1_synthetic.py dsweep                     # Cm/Cbud/Cest vs d (numpy only)
   python tier2_blackbox.py exact-nlp --K 2 --subset 10 --max_d 13 --sentences text_samples/sst2_short.txt
   python tier2b_reseed.py nlp   --backbones visobert --references zero --N 2000 --R 40 --subset 10 --sentences text_samples/sst2_samples.txt --pilot ucb
   python tier2b_reseed.py image --backbones resnet50 --references mean --N 2000 --R 40 --subset 10 --images_dir image_samples --glob "*.JPEG"
   python baselines.py nlp   --backbones distilbert --references mask --N 2000 --B 200 --subset 10 --sentences text_samples/sst2_samples.txt
   ```

---

## 7. Key results to remember (already in tables, don't recompute)

- **Tier 2 forward:** images 0/7604 flips; NLP 26/32787 (0.08%), all in visobert/zero.
- **Enumeration (R1.9):** 110 certified 2nd-order coords, 0 false signs; guarantee-margin power 35/35 = 1.000; resolution recovery 110/336 = 0.327.
- **Reseeding (R1.7):** ResNet-50/mean 0/400 per-run; ViSoBERT/zero plain 34/160 = 0.21, UCB 21/160 = 0.13 (both above 1/pK = 0.059 — flagged as limitation, NOT certified).
- **Baselines (R1.10):** naive Wald 10/10 certify-all (straw man); fair Wald 0/10; floor never certifies what fair Wald rejects.

**Honesty stance maintained throughout:** where a guarantee doesn't fully hold (visobert/zero
per-run > 1/pK even under UCB), the paper reports it as a limitation rather than overclaiming.