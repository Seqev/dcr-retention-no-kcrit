# PRE-REGISTRATION -- CROSS-MODEL RETENTION (does the 1B finding transfer, or does K_crit emerge with capacity?)

Two stacked gates: GATE A (port validity) is a HARD PRECONDITION for GATE B (cross-model science).
Rules fixed BEFORE any result; none tunable after.

## WHAT THIS DECIDES
On Llama-3.2-1B: (i) retention top-K ~ dense for c>=~0.10 up to 128K; (ii) NO absolute K_crit (k_eff inert
over [1300,5200]); (iii) the long-context limit is the BASE MODEL's own semantic disambiguation
(failure = WRONG_NEEDLE growing with N, mirrored by dense; needle IN top-k yet answer wrong:
in&wrong 0.38@32K -> 0.56@65K -> ~0.77@98K). OPEN: 1B-specific or general? Does higher capacity
(a) disambiguate better (ceiling rises, structure holds), or (b) reveal an absolute K_crit 1B was too
weak to show? Answered on a CONTROLLED capacity step.

## MODELS
  PRIMARY:   Llama-3.2-3B  (SAME family as 1B -> minimal port; clean 3x-capacity contrast).
  SECONDARY: Qwen2.5-3B    (DIFFERENT family ~3B -> tests family-specificity; ungated; clean port).
  DEFERRED:  Qwen3-14B (product/synthesis port, multihop) -- SEPARATE later gate, NOT here.

## GATE A -- PORT VALIDITY (HARD PRECONDITION; per model; run NOTHING of Gate B until ALL pass)
  A0  grep-verify attention math: n_layers, n_kv_heads, head_dim, n_rep, RoPE, AND any QK-norm
      (Qwen3 RMSNorms Q,K -> changes q-axis u_Q=Q/||Q||). Document selection formula actually used.
  A1  bf16 DENSE path bit-identical to vanilla HF: max|logit_dcr - logit_vanilla| == 0 (held-out batch).
  A2  SPARSE iso-PPL on held-out text (WikiText-2): paired delta-PPL(sparse-dense) CI CONTAINS 0 AND
      CI_upper < STRICT 0.5%, at c=0.15, N up to memory limit (A4).
  A3  Route counters: sparse decode all-DCR (sdpa=0), dense all-SDPA (dcr=0); prefill-confound re-confirmed.
  A4  EMPIRICAL memory bound: max N that fits B=1 bf16 on 16 GB; log peak VRAM per N.
  IF ANY A0-A3 fails -> STOP, report, do NOT run Gate B. A4 only BOUNDS the grid.

## GATE B -- CROSS-MODEL k_eff-perp-N (only after Gate A passes; inherit 1B machinery verbatim)
  ARMS: DENSE=enable_dcr=False (decode once); SPARSE=q_topk_triton at c yielding k_eff_target (all-DCR).
        fp32 softmax. RNG-matched prompts (hash identical dense==sparse AND across k_eff). Greedy MAX_NEW=32.
        MODE=SEMANTIC, K=16, density interval=500.
  k_eff TARGETS: {1300,2000,2600,3500,5200} -- SAME ABSOLUTE as 1B. window_floored=0.
  N GRID: {32000,65536} guaranteed; 98304/131072 only if A4 fits.
  PILOT (n=5 full grid): informative := acc_dense in [0.30,0.95]. If model ceilings at K=16 -> escalate
        K in {16,32} (pre-stated), then large n. CEILING/FLOOR cells excluded (reported).
  SAMPLE/CI: informative n>=100 (>=60 large N). Paired bootstrap 10000 -> CI on (dense-sparse) AND
        (wrong_needle_sparse - wrong_needle_dense). McNemar.

## PRIMARY READOUTS (STRUCTURE, not raw accuracy)
  R1 RETENTION penalty (dense-sparse): transfers iff CI_upper<=0.05 across informative cells.
  R2 DISSOCIATION {needle_in_topk x correct}: in&WRONG fraction per N (1B: 0.38->0.56->0.77, rises w/ N).
  R3 FAILURE MODE: WRONG_NEEDLE (mis-sel) vs STARVATION (OTHER+MISS) differential vs dense (1B: WRONG_NEEDLE-dom).
  R4 k_eff INERTNESS: does acc move with k_eff over [1300,5200]? (1B flat). If acc RISES w/ k_eff,
     sparse<dense below a threshold flat in N -> K_crit EMERGES with capacity.

## VERDICTS (per model, then synthesis)
  RETENTION-TRANSFERS = R1 holds. WEIGHTS-BOUND-UNIVERSAL = R2 present & rises w/ N & R3 WRONG_NEEDLE-dom & R4 flat.
  CAPACITY-SHIFTS-CEILING = R2 in&WRONG LOWER than 1B at matched N but structure (R1,R4) holds.
  K_CRIT-EMERGES = R4 acc rises w/ k_eff, sparse<dense below a threshold INDEPENDENT of N.

## WHAT NOT TO DO
  - No Gate B before Gate A. - Don't assume Llama selection formula transfers (grep QK-norm). - Compare
    STRUCTURE not raw acc. - No fabricated OOM/floor; no silent SDPA fallback. - No tuning after results.
  - No synthesis claim (multihop = separate gate). - Don't modify source/tests/sealed bundle; reports only.

## OUTPUT -> <report_dir>/crossmodel_retention_<ts>/
  PREREGISTRATION.md, port_validity.md (A0-A4 per model), crossmodel_results.jsonl, CROSSMODEL_PROGRESS.md,
  per_prompt_log.csv, CROSSMODEL_REPORT.md (per-model grid; R1-R4; in&WRONG-vs-N vs 1B; per-model verdict;
  synthesis; memory bounds; limitations), run.log, mechanism_recheck.txt, grep_check.md

SEED=20260531. GPU RTX 4060 Ti sm_89. Greedy, fp32 softmax. 1B = REFERENCE baseline (not re-run).

## LIMITATIONS
  - N memory-bounded on larger models (expect <=64K on 3B bf16); 1B dissociation clear by 65K so transfer
    is answerable within bound; large-N tail (>=98K) may be 1B-only. - bf16; greedy no-CoT; semantic only.
  - 2-3 models != scaling law; T3.2 (M,D,N,k_eff) needs more models (NEXT gate). - K_CRIT-EMERGES would be
    capacity-specific not universal; cross-arch confirmation still required before any "law".
