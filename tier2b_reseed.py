"""
tier2b_reseed.py
================
TIER 2b -- INDEPENDENT-RESEEDING AUDIT.

Question answered: does the PROBABILITY statement of Theorem 1 hold across
INDEPENDENT mask draws? The nested prefix ladder of Tier 2 (tier2_blackbox.py)
reuses ONE mask bank, so its sign-stability is partly built in by Corollary 2 and
cannot test the "with probability >= 1 - delta" clause. Here we fix ONE budget N
and run R INDEPENDENT seeds per probe, then measure the two things the nested
ladder cannot.

WHAT IS MEASURED (all numerics via bl_core.reseed_audit):

  (A) PER-COORDINATE cross-seed sign-disagreement rate among certified
      coordinates. Target <= 1/pK at delta = 1/pK. This is the quantity the
      pre-revision paper reported (Table 4 "viol./cert").

  (R1.7) PER-RUN, SIMULTANEOUS failure rate -- the fraction of independent draws
      in which ANY certified coordinate disagrees with the cross-seed reference
      sign. Theorem 1's failure event is per-run and simultaneous: on that event
      arbitrarily many coordinates may be wrong at once, so the faithful test of
      "with probability 1 - delta" is this item-level (any-coordinate) rate, not
      the per-coordinate rate (A). Target <= delta = 1/pK. Reported ALONGSIDE (A)
      so the two are never conflated -- exactly the referee's R1.7 point.

  (B) BAND-STRATIFIED Jaccard stability of the certified set across seeds, split
      by whether a coordinate's cross-seed median magnitude is ABOVE 2*floor or
      INSIDE the unresolved band [floor, 2*floor]. Definition 1 predicts set
      churn is confined to the band; above-band membership should be near-perfect.

STABILITY vs CORRECTNESS (R1.8). Cross-seed agreement is STABILITY, not
CORRECTNESS: two seeds can agree and both be wrong. The reference sign here is
the cross-seed MAJORITY, a proxy for sgn(beta_S), not ground truth. Only the
enumeration tier (exact beta, tier2_blackbox exact-nlp) tests correctness. This
driver labels its numbers as stability throughout.

R1.4 (Path A): every per-seed certified decision uses the realized floor
(bl.floor_from_design), Cest measured from that seed's own design -- no frozen
constant, no transfer claim.

USAGE
  # NLP, weakest-signal cell (the ViSoBERT/zero stress case):
  python tier2b_reseed.py nlp --backbones visobert --references zero \
      --N 2000 --R 40 --K 1 --subset 10 --sentences sst2_samples.txt

  # Image, strongest-signal cell (deterministic backbone):
  python tier2b_reseed.py image --backbones resnet50 --references mean \
      --N 2000 --R 40 --subset 10 --images_dir benchmark_50 --glob "*.JPEG"

  # Pure-numpy pipeline self-test (no models, no downloads):
  python tier2b_reseed.py selftest
"""
from __future__ import annotations
import argparse
import glob
import math
import os
import sys
import numpy as np

import bl_core as bl
# Reuse the Tier-2 probe adapters / pilot / progress so both tiers share ONE
# definition of a probe and of the pilot sigma_eff (no divergence between tiers).
from tier2_blackbox import (
    Progress, Probe, nlp_probe, image_probe, pilot_sigma_eff, load_sentences,
)


# =========================================================================== #
#  Per-probe audit
# =========================================================================== #
def audit_probe(probe: Probe, N, R, K, seed0=0, ucb=False):
    """Pilot sigma_eff ONCE (as deployed), then run the independent-reseeding
    audit at fixed N over R seeds. Returns a bl.ReseedResult or None if the
    probe is not identifiable at this (N, K).

    R2.1: ucb=True uses the empirical-Bernstein UPPER-CONFIDENCE sigma_eff
    (unconditional guarantee), and the floor then budgets the pilot event as the
    fourth union-bound term (split=4). ucb=False is the plain conditional pilot
    (split=3, the default)."""
    if N <= bl.p_K(probe.d, K):
        return None
    # Pilot once (as deployed); grab the mismatch breakdown so the per-seed floor
    # is the honest TWO-TERM certified floor (report point 1/2). detail=True gives
    # the MismatchEstimate (plain) or SigmaEffUCB (ucb).
    s_eff, est = pilot_sigma_eff(probe, K, seed=seed0, ucb=ucb, detail=True)
    if ucb:
        m_for_floor, B_for_floor = est.m_ucb, getattr(probe, "B_known", None)
    else:
        m_for_floor, B_for_floor = est.m_hat, est.B_hat
    split = 4 if ucb else None
    res = bl.reseed_audit(
        query_fn=probe.query, d=probe.d, sigma_obs=probe.sigma_obs,
        N=N, R=R, K=K, s_eff=s_eff, seed0=seed0 + 1000, split=split,
        m_hat=m_for_floor, B=B_for_floor)
    return res if (res is not None and res.well_posed) else None


# =========================================================================== #
#  Aggregation across probes within a cell
# =========================================================================== #
def _pool_cell(cell, results):
    """Pool per-probe ReseedResults into one cell row.

    Per-coordinate and per-run rates are POOLED over the numerators/denominators
    of every probe in the cell (so a cell rate is the honest fraction, not a mean
    of fractions). Jaccard and band counts are averaged over well-posed probes.
    """
    n_cc = sum(r.n_cert_checks for r in results)
    n_vi = sum(r.n_viol for r in results)
    n_ru = sum(r.n_runs for r in results)
    n_rf = sum(r.n_run_fail for r in results)
    pk = int(np.median([r.pK for r in results]))
    def _mn(xs):
        xs = [x for x in xs if x is not None and not math.isnan(x)]
        return float(np.mean(xs)) if xs else float("nan")
    return {
        "cell": cell,
        "pK": pk,
        "target": 1.0 / pk,
        "n_probes": len(results),
        "viol": n_vi, "cert": n_cc,
        "viol_rate": (n_vi / n_cc) if n_cc else float("nan"),
        "run_fail": n_rf, "runs": n_ru,
        "run_fail_rate": (n_rf / n_ru) if n_ru else float("nan"),
        "jac_above": _mn([r.jaccard_above for r in results]),
        "jac_band": _mn([r.jaccard_band for r in results]),
        "n_above": _mn([r.n_above for r in results]),
        "n_band": _mn([r.n_band for r in results]),
    }


# =========================================================================== #
#  Reporting
# =========================================================================== #
def report_reseed(rows, N, R, K, pilot="plain"):
    print("\n" + "=" * 72)
    _plabel = ("UCB upper-confidence sigma_eff (R2.1): discharges Theorem 1's "
               "pilot\n                  conditioning hypothesis w.h.p. (split=4). "
               "Does NOT by itself force\n                  the per-run rate below "
               "1/pK -- a near-degenerate cell can remain in\n                  the "
               "residual conditional regime the paper flags as a limitation."
               if pilot == "ucb"
               else "plain point-estimate sigma_eff -- conditional (split=3)")
    print(f"TIER 2b -- INDEPENDENT-RESEEDING AUDIT  (fixed N={N}, R={R} seeds, "
          f"K={K})")
    print(f"           pilot: {_plabel}")
    print("=" * 72)
    if not rows:
        print("  No cell produced a result. Likely causes, in order:")
        print("   (1) no input files matched (--images_dir/--glob or --sentences"
              " empty/mismatched path);")
        print("   (2) every probe's d put N <= pK at this budget (raise --N or"
              " lower d);")
        print("   (3) an ill-conditioned design at every seed (no certificate).")
        print("  Check the per-item progress log above: '(skipped)' => (1),"
              " '(not identifiable)' => (2)/(3).")
        return
    print(f"  {'cell':>20} {'pK':>4} | {'viol/cert':>12} {'rate':>7} "
          f"{'1/pK':>7} | {'run_fail/runs':>14} {'rate':>7} | "
          f"{'J>2fl':>6} {'J[fl,2fl]':>9} {'#>2fl/#band':>12}")
    for r in rows:
        vc = f"{r['viol']}/{r['cert']}"
        rf = f"{r['run_fail']}/{r['runs']}"
        print(f"  {r['cell']:>20} {r['pK']:>4} | "
              f"{vc:>12} {r['viol_rate']:>7.4f} {r['target']:>7.4f} | "
              f"{rf:>14} {r['run_fail_rate']:>7.4f} | "
              f"{r['jac_above']:>6.3f} {r['jac_band']:>9.3f} "
              f"{r['n_above']:>5.1f}/{r['n_band']:<5.1f}")

    # Guarantee-level summary (pooled across every cell).
    tot_vi = sum(r["viol"] for r in rows)
    tot_cc = sum(r["cert"] for r in rows)
    tot_rf = sum(r["run_fail"] for r in rows)
    tot_ru = sum(r["runs"] for r in rows)
    tgt = float(np.median([r["target"] for r in rows]))
    pc_rate = (tot_vi / tot_cc) if tot_cc else float("nan")
    rn_rate = (tot_rf / tot_ru) if tot_ru else float("nan")
    print("\n  (A) PER-COORDINATE cross-seed sign-disagreement (stability, "
          "not correctness):")
    print(f"      {tot_vi}/{tot_cc} = {pc_rate:.4f}   target <= 1/pK "
          f"(~{tgt:.4f})   [{'OK' if pc_rate <= tgt else 'ABOVE TARGET'}]")
    print("  (R1.7) PER-RUN simultaneous failure (ANY certified coord "
          "disagrees) -- the faithful test of Theorem 1's 1-delta clause:")
    print(f"      {tot_rf}/{tot_ru} = {rn_rate:.4f}   target <= delta = 1/pK "
          f"(~{tgt:.4f})   [{'OK' if rn_rate <= tgt else 'ABOVE TARGET'}]")
    ja = float(np.nanmean([r["jac_above"] for r in rows]))
    jb = float(np.nanmean([r["jac_band"] for r in rows]))
    print("  (B) set churn is confined to the unresolved band (Definition 1):")
    print(f"      mean Jaccard above 2*floor = {ja:.3f}  (near 1 = stable); "
          f"in band [floor,2floor] = {jb:.3f}  (lower = churn, as predicted)")
    print("\n  NOTE: cross-seed agreement is STABILITY, not correctness -- two "
          "seeds\n        can agree and both be wrong. Only the enumeration tier "
          "(exact\n        beta) tests correctness.")
    _latex_table(rows, N, R, K)


def _latex_table(rows, N, R, K):
    print("\n  % ---- booktabs table for the appendix ----")
    print("  \\begin{tabular}{lrrrrrr}")
    print("  \\toprule")
    print("  Cell & $p_K$ & viol/cert & per-run & $1/p_K$ & "
          "$J_{>2\\mathrm{fl}}$ & $J_{[\\mathrm{fl},2\\mathrm{fl}]}$ \\\\")
    print("  \\midrule")
    for r in rows:
        print(f"  {r['cell']} & {r['pK']} & {r['viol_rate']:.4f} & "
              f"{r['run_fail_rate']:.4f} & {r['target']:.4f} & "
              f"{r['jac_above']:.3f} & {r['jac_band']:.3f} \\\\")
    print("  \\bottomrule")
    print("  \\end{tabular}")


# =========================================================================== #
#  NLP / image drivers
# =========================================================================== #
def run_nlp(args):
    import bl_models as M
    backbones = (list(M.NLP_BACKBONES) if args.backbones == "all"
                 else args.backbones.split(","))
    refs = (list(M.NLP_REFERENCES) if args.references == "all"
            else args.references.split(","))
    sents = load_sentences(args.sentences, args.subset)
    rows = []
    prog = Progress(len(backbones) * len(refs) * len(sents),
                    f"tier2b-nlp N={args.N} R={args.R}")
    for bk in backbones:
        try:
            clf = M.TextClassifier(model=bk, dataset=args.dataset)
        except Exception as e:
            prog.note(f"[skip {bk}] {e}")
            continue
        for ref in refs:
            cell_results = []
            for si, sent in enumerate(sents):
                p = nlp_probe(clf, sent, ref, args.max_free)
                if p is None:
                    prog.step(f"{bk}/{ref} (skipped: d out of range)")
                    continue
                res = audit_probe(p, args.N, args.R, args.K, seed0=si, ucb=args.pilot=="ucb")
                if res is not None:
                    cell_results.append(res)
                    prog.step(f"{bk}/{ref} d={p.d} "
                              f"viol={res.n_viol}/{res.n_cert_checks} "
                              f"runfail={res.n_run_fail}/{res.n_runs}")
                else:
                    prog.step(f"{bk}/{ref} d={p.d} (not identifiable)")
            if cell_results:
                rows.append(_pool_cell(f"{bk}/{ref}", cell_results))
        clf.close()
    prog.close()
    report_reseed(rows, args.N, args.R, args.K, args.pilot)


def run_image(args):
    import bl_models as M
    paths = sorted(glob.glob(os.path.join(args.images_dir, args.glob)))
    paths = paths[:args.subset] if args.subset else paths
    bks = (list(M.IMAGE_BACKBONES) if args.backbones == "all"
           else args.backbones.split(","))
    rfs = (list(M.IMAGE_REFERENCES) if args.references == "all"
           else args.references.split(","))
    rows = []
    prog = Progress(len(bks) * len(rfs) * max(len(paths), 1),
                    f"tier2b-image N={args.N} R={args.R}")
    for bk in bks:
        try:
            clf = M.ImageClassifier(backbone=bk)
        except Exception as e:
            prog.note(f"[skip {bk}] {e}")
            continue
        for ref in rfs:
            cell_results = []
            for pi, path in enumerate(paths):
                img = clf.load_image(path)
                p = image_probe(clf, img, ref, args.grid)
                if p is None:
                    prog.step(f"{bk}/{ref} (skipped)")
                    continue
                res = audit_probe(p, args.N, args.R, 1, seed0=pi, ucb=args.pilot=="ucb")
                if res is not None:
                    cell_results.append(res)
                    prog.step(f"{bk}/{ref} d={p.d} "
                              f"viol={res.n_viol}/{res.n_cert_checks} "
                              f"runfail={res.n_run_fail}/{res.n_runs}")
                else:
                    prog.step(f"{bk}/{ref} (not identifiable)")
            if cell_results:
                rows.append(_pool_cell(f"{bk}/{ref}", cell_results))
        clf.close()
    prog.close()
    report_reseed(rows, args.N, args.R, 1, args.pilot)


# =========================================================================== #
#  Self-test: pure-numpy synthetic probe, no models, no downloads
# =========================================================================== #
def _make_synthetic_probe(d, n_active, m_resid, sigma_obs, seed, amp_lo=0.15,
                          amp_hi=0.5, degenerate=False):
    """A model-free Probe whose response is a known degree-1 Walsh signal plus a
    controlled higher-order mismatch block plus optional query noise, exactly the
    Tier-1 generator -- so the audit can be exercised end-to-end without torch.

    amp_lo/amp_hi set the active-coefficient magnitude relative to the floor:
    small amplitudes put coordinates in the unresolved band and produce the churn
    Definition 1 predicts. `degenerate` shrinks the whole response toward zero
    (the ViSoBERT/zero stress case) so the pilot can under-estimate sigma_eff and
    the per-coordinate rate can approach 1/pK.
    """
    rng = np.random.default_rng(seed)
    beta = np.zeros(d)
    active = rng.choice(d, size=min(n_active, d), replace=False)
    beta[active] = rng.choice([-1.0, 1.0], size=len(active)) * \
        rng.uniform(amp_lo, amp_hi, size=len(active))
    if degenerate:
        beta *= 0.15                       # near-constant response
    trip = rng.choice(d, size=min(3, d), replace=False)
    amp = math.sqrt(max(m_resid, 0.0))

    def query(Z):
        Zc = 2.0 * (Z - 0.5)
        main = Zc @ beta
        hi = amp * np.prod(Zc[:, trip], axis=1) if len(trip) == 3 else 0.0
        noise = (sigma_obs * np.random.default_rng().standard_normal(Z.shape[0])
                 if sigma_obs > 0 else 0.0)
        return main + hi + noise

    return Probe(d=d, target=0, query_fn=query, sigma_obs=sigma_obs), beta


def run_selftest(args):
    print("=" * 72)
    print("TIER 2b SELF-TEST -- pure numpy, no models. Exercises the full "
          "reseed\n           pipeline on synthetic probes with known structure.")
    print("           Expected: strong -> stable; near-floor -> band churn;")
    print("                     degenerate -> per-run rate rises toward 1/pK.")
    print("=" * 72)
    cells = [
        # strong signal, low noise: everything above 2*floor, near-perfect
        ("synth/strong",    dict(d=12, n_active=4, m_resid=0.02, sigma_obs=0.02,
                                 amp_lo=0.30, amp_hi=0.60)),
        # near-floor amplitudes: coords land in the unresolved band -> J_band < 1
        ("synth/nearfloor", dict(d=12, n_active=6, m_resid=0.05, sigma_obs=0.05,
                                 amp_lo=0.05, amp_hi=0.14)),
        # degenerate (ViSoBERT/zero analogue): near-constant response, pilot
        # strained -> per-coordinate / per-run rate approaches 1/pK
        ("synth/degenerate", dict(d=12, n_active=4, m_resid=0.08, sigma_obs=0.02,
                                  amp_lo=0.10, amp_hi=0.20, degenerate=True)),
    ]
    rows = []
    for name, cfg in cells:
        results = []
        for si in range(args.subset or 8):
            probe, _ = _make_synthetic_probe(seed=si, **cfg)
            res = audit_probe(probe, args.N, args.R, 1, seed0=si, ucb=args.pilot=="ucb")
            if res is not None:
                results.append(res)
        if results:
            rows.append(_pool_cell(name, results))
    report_reseed(rows, args.N, args.R, 1, args.pilot)


# =========================================================================== #
#  CLI
# =========================================================================== #
def build_parser():
    p = argparse.ArgumentParser(description="Tier 2b independent-reseeding audit")
    p.add_argument("mode", choices=["nlp", "image", "selftest"])
    p.add_argument("--N", type=int, default=2000, help="single fixed budget")
    p.add_argument("--R", type=int, default=40, help="number of independent seeds")
    p.add_argument("--K", type=int, default=1)
    p.add_argument("--backbones", default="all")
    p.add_argument("--references", default="all")
    p.add_argument("--dataset", default="sst2")
    p.add_argument("--sentences", default="sst2_samples.txt")
    p.add_argument("--max_free", type=int, default=40)
    p.add_argument("--images_dir", default="benchmark_50")
    p.add_argument("--glob", default="*.JPEG")
    p.add_argument("--grid", type=int, default=7)
    p.add_argument("--subset", type=int, default=10)
    p.add_argument("--pilot", choices=["plain", "ucb"], default="plain",
                   help="plain = conditional point-estimate pilot (split=3); "
                        "ucb = R2.1 empirical-Bernstein upper-confidence "
                        "sigma_eff, unconditional guarantee (split=4).")
    return p


def main():
    args = build_parser().parse_args()
    if args.mode == "nlp":
        run_nlp(args)
    elif args.mode == "image":
        run_image(args)
    else:
        run_selftest(args)


if __name__ == "__main__":
    main()