# PRE-REGISTRATION — k_eff ⊥ N DECOUPLING (test the Critical Semantic Capacity "Law")

Written BEFORE any result. k_eff formula grep-verified (Step 0.5, carried from c-sweep). Rules fixed; none tunable after.

## WHAT THIS GATE DECIDES (and why the c-sweep could NOT)
The c-sweep found a boundary ONLY at (c=0.02, N=131072), where k_eff=2622. From ONE (c,N) point one
CANNOT distinguish three rival hypotheses, all consistent with that single FAIL:
  H_abs : failure onsets at an ABSOLUTE k_eff (~2600) regardless of N   (the claimed "K_crit law")
  H_c   : failure onsets at a COVERAGE ratio (c ~ 0.02) regardless of N
  H_N   : failure onsets at a CONTEXT LENGTH (N ~ 128K) regardless of k_eff
c and N were CONFOUNDED in the c-sweep (c=0.02 AND N=128K moved together). This gate DECOUPLES them:
hold k_eff at a fixed ABSOLUTE value and walk N. Only this separates H_abs from H_c / H_N.
SECONDARY: push near-dup aliasing with larger n to move it OUT of the c-sweep's borderline.

## CLAIM UNDER TEST (pre-stated, falsifiable)
"Critical Semantic Capacity Law", Regime I: there exists an absolute K_crit (~2600 for Llama-3.2-1B)
such that for k_eff > K_crit retention holds INDEPENDENT OF N.
  CONFIRM  if the FAIL onset tracks a fixed ABSOLUTE k_eff across N (vertical line in (k_eff,N)) --
           same k_eff breaks at every N, larger k_eff holds at every N.
  FALSIFY  if the onset tracks c (a c-line) or N (an N-line) instead -- then "independent of N" is FALSE
           and the law is reduced to "c-dependent" or "N-dependent". A FALSIFY here is a SUCCESS of the
           test: it stops the law being written as fact.

## DESIGN -- the decoupling (the whole point)
Drive k_eff DIRECTLY (absolute key budget), NOT via c. k_eff is set per cell to a TARGET absolute value
by choosing c = k_eff_target / N per (cell, N) (then verify the grep formula reproduces it; log actual
k_eff and assert |actual - target| <= 2 after even-rounding; window_floored must be 0, i.e. target>64).

  ABSOLUTE k_eff targets: {1300, 2000, 2600, 3500, 5200}  (straddle the claimed ~2600 onset, ~2x span).
  N grid: {32000, 65536, 131072, 196608}  (196608 = 1.5x the only N that failed; tests "independent of N"
    on the LARGE side too. If 196608 OOMs at decode -> report, drop it, do NOT fabricate.)
  => for each (k_eff_target, N) compute c = k_eff_target / N; SKIP any (k_eff,N) where c > 1.0 or
     k_eff_target >= N (degenerate / dense) -> tag "n/a", exclude.
  MODE: SEMANTIC primary (this is where the c-sweep boundary lived, starvation-dominated).
  CELLS / competition: K=16 (highest competition from c-sweep), density high (interval=500), as before.

## SECONDARY BRANCH -- near-dup aliasing (resolve the borderline)
The c-sweep's only MIS-SELECTION signal was neardup, and it was BORDERLINE (CI barely). Re-run NEARDUP,
K=16, at the SAME absolute k_eff targets, N in {32000, 131072}, with n>=200 (double) to move wrong-needle
out of borderline. This isolates the dot-product-aliasing failure mode (claimed Regime II) from the
semantic-starvation mode (Regime III). They are likely DIFFERENT mechanisms on DIFFERENT axes -- do not
merge them.

## ARMS / CONTROL (inherit verbatim)
  DENSE=enable_dcr=False (all-SDPA, c-INDEPENDENT -> decode ONCE per prompt);
  SPARSE=q_topk_triton at the c that yields k_eff_target (all-DCR). fp32 softmax (INS-28).
  RNG-matched prompts (per-prompt seed; hash asserted identical dense==sparse AND identical across all
  k_eff within a (mode,N) cell). Greedy, MAX_NEW=32. Route counters per cell. Prefill-confound re-confirmed once.

## PROMPT IDENTITY ACROSS k_eff + DENSE-ONCE
  Within each (mode,N) cell the SAME prompt set at EVERY k_eff target. dense decoded ONCE per prompt;
  sparse decoded at each k_eff on the SAME prompts => PAIRED PER PROMPT across k_eff. Per-prompt
  k_eff-onset logged (the k_eff at which a prompt flips correct->non-correct). prompt_hash identical (asserted).

## THREE-WAY FAILURE ATTRIBUTION (carry from c-sweep; load-bearing here)
  WRONG-NEEDLE = first_code == a different planted code (MIS-SELECTION; top-k took the wrong key) [Regime II].
  STARVATION   = OTHER-CODE (hallucinated, no plant) + MISS (no complete code) (too few keys; degenerate) [Regime III].
  Report differentials vs dense separately, each with paired CI:
    (wrong_needle_sparse - wrong_needle_dense) AND (starvation_sparse - starvation_dense).
  The MECHANISM at onset MUST be named per cell (mis-selection vs starvation). The c-sweep found onset
  is STARVATION-dominated in semantic; CONFIRM or REVISE that here. Do NOT assert "dot-product
  unresolvability" (Regime II mechanism) for a STARVATION onset -- that was the inversion in the draft theorem.

## NEEDLE-IN-TOPK (carry from c-sweep; the leak guard)
  needle_in_topk per prompt per k_eff at the answer-emitting step (q-axis score rank of target needle
  span < k_eff; mean over (layer,head); bool = mean>=0.5). Report rate AND joint {needle_in_topk x correct}.
  A cell that "passes" while needle_in_topk has dropped is ROBUST-BUT-SUSPECT (answer via leak/redundancy,
  Regime III dissociation), NOT a clean retention pass. Distinguish in the verdict.

## SCORING  first complete 7-digit code = committed answer; trichotomy + OTHER-CODE; wrong-needle never -> miss.

## SAMPLE / CI  n>=100 (>=60 at N>=131072 if VRAM/time forces; actual n reported); neardup branch n>=200.
  Paired bootstrap (10000) -> 95% CI on (acc_dense_correct - acc_sparse_correct@k_eff) AND
  (wrong_needle_sparse@k_eff - wrong_needle_dense). McNemar b/c reported.
  FAIL_cell@k_eff if CI_upper(dense-sparse correct) > 0.05 OR CI_upper(wrong_sparse - wrong_dense) > 0.05.

## VALIDITY
  dense in [0.30,0.95] per (mode,N) cell (c-independent; confirm once/cell; if dense ceilings/floors at a
  given N the cell is non-informative -> excluded, reported). window_floored must be 0 (target>64); else exclude.

## PRIMARY VERDICT -- which axis owns the boundary (monotone, fixed pre-hoc)
  For each fixed k_eff_target, read FAIL across N. For each fixed N, read FAIL across k_eff.
  - LAW-CONFIRMED (H_abs) = FAIL onset is at a fixed ABSOLUTE k_eff, the SAME across N (within one
    k_eff step), AND larger k_eff holds across N, AND smaller k_eff fails across N (monotone in k_eff,
    flat in N). Report K_crit as the onset k_eff with its across-N consistency.
  - C-DEPENDENT (H_c) = onset tracks c = k_eff/N (fails at a fixed RATIO, not a fixed count) -> the "law"
    is a coverage law; "independent of N" FALSIFIED.
  - N-DEPENDENT (H_N) = onset tracks N (large N fails even at large k_eff) -> length-driven; "independent
    of N" FALSIFIED; points to a positional / long-context effect, not a key-budget law.
  - MIXED = onset surface depends on both -> report the (k_eff,N) boundary surface; no single scalar law.
  A FALSIFY (C-/N-/MIXED) is a valid, publishable result: it correctly bounds the claim.

## SECONDARY VERDICT -- aliasing (neardup branch)
  ALIASING-REAL = neardup wrong_needle differential CI EXCLUDES 0 at some k_eff (with n>=200) AND grows as
    k_eff shrinks -> Regime II (mis-selection) is a genuine distinct mode; report its onset k_eff.
  ALIASING-NOT-ESTABLISHED = CI still contains 0 at n>=200 -> the c-sweep's borderline was noise; Regime II
    is NOT supported; the only established failure mode is starvation (Regime III).

## CONTINUITY (MECHANICAL)
  At the (k_eff, N) point closest to the c-sweep's FAIL (k_eff~2600, N=131072) the result must reproduce
  the c-sweep (paired CI of (this - c-sweep) for acc_correct CONTAINS 0). If it excludes 0 -> STOP, report drift.

## INCREMENTAL CHECKPOINTS
  Order: semantic primary, k_eff ascending within each N, N ascending. After each (cell,k_eff): append
  JSONL, rewrite PROGRESS.md (running table incl. c, k_eff actual, window_floored, needle_in_topk, both
  differentials + CIs, FAIL, mechanism), copy live. Per-prompt rows live. Log wallclock + VRAM; watch OOM
  at N=131072/196608 (report, never fake).

## WHAT NOT TO DO
  - Do NOT vary c and read it as k_eff -- k_eff is the driven variable; c is derived. Log BOTH.
  - Do NOT call a STARVATION onset "dot-product unresolvability"/Regime-II -- name the mechanism the
    3-way attribution actually shows (this was the draft theorem's inversion).
  - Do NOT call the smooth k_eff degradation a "phase transition" unless the data show a sharp threshold
    (an abrupt FAIL between adjacent k_eff steps with flat behaviour either side) -- report the shape honestly.
  - Do NOT report K_crit as N-independent unless the across-N flatness is actually observed (the whole gate).
  - Do NOT tune band/threshold/CI/monotone after results. Do NOT collapse wrong-needle into miss.
  - Do NOT let silent SDPA fallback pass as sparse. Do NOT fabricate OOM/excluded cells.
  - Do NOT modify source / tests / sealed bundle. Reports to the report dir only; non-destructive.
  - Step 0: verify report dir writable; else STOP. Step 0.5: re-grep k_eff formula, assert unchanged.

## OUTPUT -> <report_dir>/keff_perp_N_<ts>/
  PREREGISTRATION.md (this, first), keff_results.jsonl, KEFF_PROGRESS.md, per_prompt_log.csv
  (mode,N,K,k_eff_target,k_eff_actual,c,sample,arm,outcome,first_code,needle_in_topk,keff_onset,prompt_hash,oom),
  KEFF_REPORT.md (the (k_eff x N) FAIL grid for semantic; the neardup aliasing table; primary verdict
  H_abs/H_c/H_N/MIXED with the boundary surface; secondary aliasing verdict; needle-in-topk joint;
  continuity check; mechanism-at-onset per cell; honest shape (phase-transition vs smooth); limitations),
  run.log, mechanism_recheck.txt, kff_grep_check.txt

SEED=20260531 (== v3/c-sweep, prompts identical where N matches). Model meta-llama/Llama-3.2-1B (cached).
GPU RTX 4060 Ti sm_89. Greedy, fp32 softmax (INS-28).

## LIMITATIONS (pre-stated)
  - K_crit, if confirmed, is for Llama-3.2-1B only; cross-model is a separate gate (it may scale with
    d_model / n_heads / head_dim -- note as the natural follow-up, do NOT assume).
  - k_window=64 fixed; all targets are >64 (above the recency floor) by construction.
  - greedy no-CoT; semantic + neardup retrieval only (multihop floors on 1B). A capacity law for RETRIEVAL
    selection is NOT a law for synthesis.
  - N=196608 added to test the large-N side of "independent of N"; if it OOMs, the across-N claim is
    bounded to <=131072 (report the bound honestly).
  - "Law" language is provisional until across-N flatness is observed; absent that, report a boundary
    SURFACE, not a scalar law.
