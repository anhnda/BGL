# RUN COMMANDS — final code/UCB revision

Run all commands from the repository directory containing `bl_core.py`.

## 0. Replace the changed files

Changed files in this patch:
- `bl_core.py`
- `tier2_blackbox.py`
- `tier2b_reseed.py`
- `baselines.py`
- `bl_models.py`

## 1. Syntax + pure-numpy smoke tests

```bash
python -m py_compile *.py visualize/*.py

python tier2b_reseed.py selftest \
  --N 300 --R 6 --subset 2 --pilot plain

python tier2b_reseed.py selftest \
  --N 300 --R 6 --subset 2 --pilot ucb
```

Expected:
- no syntax/runtime error;
- item-level denominator includes every well-posed draw, including empty-certified-set draws;
- UCB mode says `mismatch-energy UCB`, not `sigma_eff UCB`;
- `1/pK` is printed only as a nominal reference scale.

## 2. Regenerate Table 4 — required

Use the exact same frozen item lists/seeds as the manuscript.

Set your paths first, for example:

```bash
export VSFC_SENTENCES=/path/to/the/exact_vsfc_10_items.txt
export IMAGE_DIR=/path/to/benchmark_50
```

### 2a. ViSoBERT / zero — plain row

```bash
python tier2b_reseed.py nlp \
  --backbones visobert \
  --references zero \
  --dataset vsfc \
  --sentences "$VSFC_SENTENCES" \
  --N 2000 --R 40 --K 1 --subset 10 \
  --pilot plain \
  | tee table4_visobert_zero_plain.log
```

### 2b. ViSoBERT / zero — NEW UCB row

```bash
python tier2b_reseed.py nlp \
  --backbones visobert \
  --references zero \
  --dataset vsfc \
  --sentences "$VSFC_SENTENCES" \
  --N 2000 --R 40 --K 1 --subset 10 \
  --pilot ucb \
  | tee table4_visobert_zero_ucb.log
```

For this NLP probability-output case the code automatically uses:
- `sigma_obs_ub = 0` (deterministic `.eval()` inference),
- direct `m_UCB` from held-out squared residuals,
- `delta_pilot = delta/4`,
- `split = 4`,
- `B_pop_ub = 1 + sqrt(pK)` from the generic bounded-output Walsh/Parseval bound.

### 2c. ResNet-50 / mean — plain stability row

```bash
python tier2b_reseed.py image \
  --backbones resnet50 \
  --references mean \
  --images_dir "$IMAGE_DIR" \
  --glob "*.JPEG" \
  --N 2000 --R 40 --subset 10 \
  --pilot plain \
  | tee table4_resnet50_mean_plain.log
```

Do NOT run strict `--pilot ucb` on image logits unless you have a justified finite
population residual bound. If you do have one:

```bash
python tier2b_reseed.py image \
  --backbones resnet50 \
  --references mean \
  --images_dir "$IMAGE_DIR" \
  --glob "*.JPEG" \
  --N 2000 --R 40 --subset 10 \
  --pilot ucb \
  --B_pop_ub <VALID_BOUND>
```

Never use the sample residual maximum as `B_pop_ub`.

## 3. Re-run exact enumeration — required

This changed because:
1. every run-certified coordinate is now scored, including a true-zero exact coefficient;
2. the run floor uses exact full-cube `m_{>K}` and exact `B_pop`.

For SST-2 backbones:

```bash
export SST2_SENTENCES=/path/to/the/exact_sst2_items.txt

python tier2_blackbox.py exact-nlp \
  --K 2 --subset 10 --max_d 13 \
  --backbones distilbert,roberta \
  --references mask,pad,zero \
  --dataset sst2 \
  --sentences "$SST2_SENTENCES" \
  | tee exact_nlp_sst2_final.log
```

For ViSoBERT / VSFC:

```bash
python tier2_blackbox.py exact-nlp \
  --K 2 --subset 10 --max_d 13 \
  --backbones visobert \
  --references mask,pad,zero \
  --dataset vsfc \
  --sentences "$VSFC_SENTENCES" \
  | tee exact_nlp_vsfc_final.log
```

Update Section 5.3 / reviewer response with the newly printed:
- false signs / scored,
- factor-two guarantee power,
- full-cube resolution recovery.

## 4. Re-run NLP part of Table 3 — recommended/required for exact sync

NLP inference is now explicitly deterministic (`sigma_obs = 0`) and the code no
longer labels probability outputs as stochastic query noise.

SST-2:

```bash
python tier2_blackbox.py nlp \
  --K 1 \
  --backbones distilbert,roberta \
  --references mask,pad,zero \
  --dataset sst2 \
  --beta_min 0.02 \
  --N_ladder 512,1000,2000,4000 \
  --subset 10 \
  --sentences "$SST2_SENTENCES" \
  | tee table3_nlp_sst2_final.log
```

ViSoBERT / VSFC:

```bash
python tier2_blackbox.py nlp \
  --K 1 \
  --backbones visobert \
  --references mask,pad,zero \
  --dataset vsfc \
  --beta_min 0.02 \
  --N_ladder 512,1000,2000,4000 \
  --subset 10 \
  --sentences "$VSFC_SENTENCES" \
  | tee table3_nlp_vsfc_final.log
```

## 5. What does NOT need a rerun solely because of the UCB patch

- Table 2 synthetic `C_M`, `C_BUDGET`, `C_est` sweep: unchanged by the new UCB.
- Tier-1b design-only `C_est` transfer study: unchanged.
- Image Table 3: numerical path is unchanged by the UCB patch, assuming the
  augmented-Gram results in the current manuscript were already generated from
  this code generation.
- Appendix-D baselines: no numerical change from the UCB patch itself; rerun only
  if you want a clean full reproducibility pass.

## 6. Grep audit before updating LaTeX

```bash
grep -R -n \
  -e "sigma_eff_ucb" \
  -e "SigmaEffUCB" \
  -e "unconditional guarantee" \
  -e "faithful test of Theorem 1" \
  -e "run_fail_rate" \
  -- *.py
```

Expected: no old UCB/theorem wording, except backward-compat property names if you
choose to keep them.

## 7. Numbers that must be updated in LaTeX after the runs

At minimum:
- Table 4 ViSoBERT/zero plain denominator/rate;
- Table 4 ViSoBERT/zero UCB row;
- Table 4 ResNet-50/mean item-level denominator if it changes;
- Section 5.3 exact-enumeration false-sign / power / recovery counts;
- NLP Table 3 checks if the deterministic-noise wording/run changes them;
- the corresponding numbers in `response_reviewer1_revised.tex` and
  `response_reviewer2_revised.tex`.

Do not copy the old `0/668`, `0/199`, `8/256`, etc. into the final paper unless
the new runs reproduce them.
