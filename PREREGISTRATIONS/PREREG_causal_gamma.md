# PRE-REGISTRATION -- IS ATTENTION COMPUTATION OR TRANSPORT? (needle-gain gamma-sweep, v2)

Written BEFORE results. Rules/grid/metrics/verdict fixed; none tunable after.

## QUESTION
k_eff-perp-N: failure dominated by needle-IN-top-k-yet-WRONG, k_eff-inert. OPEN: is attention CAUSAL for
correctness, or only an INDICATOR of what the weights decided? Manipulate ONLY the needle's attention WEIGHT
within a FIXED selected set (the one lever no prior gate touched).
  TRANSPORT -> failure downstream of attention; closes the attention-redirection memory class; only WEIGHTS survive.
  COMPUTATION -> small needle-gain systematically shifts P(correct) & recovers argmax, needle-specifically,
    beyond tell-it -> attention participates in the decision -> flow-redirection memory class is LIVE.

## TARGET POPULATION (P1)
Semantic K=16 sparse, c=0.15 anchor. P1 = needle_in_topk==True AND outcome!=CORRECT. PRIMARY N=65536,
SECONDARY N=32000. SAME prompts/seed as k_eff-perp-N (SEED=20260531). P1 RECOMPUTED per model, never inherited.

## PHASE 0 (cheap, direction-resolving): n>=30 P1 @65536, gamma in {1,4,16}. 1B FIRST (sealed, no port).
Escalate per model only if ambiguous: full gamma {1,2,4,8,16} + Arm1b + Arm3 + step-locus + rank-binning.

## METRICS (CONTINUOUS first)
Per gamma per prompt: P(correct first token) -> Delta_P=P@gamma-P@1; logit(correct); argmax 3-way.
  REGIME-A (no effect): Delta_P~0 (CI contains 0).
  REGIME-B (sub-threshold): Delta_P>0 (CI excludes 0) but argmax rarely flips.
  REGIME-C (causal recover): Delta_P>0 AND argmax flips WRONG->CORRECT.

## THREE-WAY ARGMAX (previously-WRONG cases)
  RECOVERED=WRONG_NEEDLE->CORRECT ; DEGENERATED=WRONG_NEEDLE->OTHER/MISS ; UNMOVED=stays WRONG_NEEDLE.

## ARM 1 needle-gain: at answer step, multiply needle span's attention LOGITS by gamma BEFORE softmax
(add log gamma), selected SET identical. Report saturation shape of Delta_P(gamma).
## ARM 1b competitor-gain control: boost the WINNING-WRONG code's span instead. If it moves dynamics as much
as needle-gain -> system merely sensitive to ANY nudge -> NOT needle-specific. CAUSAL requires needle helps,
competitor does NOT (ideally competitor pushes toward wrong). Phase 0: gamma=16 only.
## ARM 2 set-ablation (ESCALATION): composition lever (which keys) vs weight (how much).
## ARM 3 information-matched tell-it (ESCALATION): append correct code in plain text. CAUSAL requires Arm1
beats baseline AND not fully explained by Arm3.
## STEP-LOCUS PILOT (ESCALATION): gain at last / last-4 / last-8 steps -- guards against false "last-step inert".
## RANK-DEPENDENCE (the regime law): pre-intervention log rank(needle) & attention_mass(needle); report effect
binned by baseline rank {1-5,6-20,21-40,>40}. Effect only in marginal zone -> regime law, not binary.

## VERDICT (mechanical)
  ATTENTION-COMPUTATION = Delta_P>0 with gamma (CI excl 0) AND needle-specific (Arm1b doesn't match) AND
    (escalation) not explained by Arm3. Sub-cases REGIME-C (argmax flips) / REGIME-B (sub-threshold).
  ATTENTION-TRANSPORT = Delta_P~0 at all loci & ranks, OR all movement DEGENERATED, OR competitor matches, OR
    Arm3 fully explains -> attention-redirection class CLOSED; only WEIGHTS survive.
  MIXED/REGIME-DEPENDENT = causal only in some rank bin / locus / gamma range.

## SETUP
meta-llama/Llama-3.2-1B (cached, sealed); greedy, fp32 softmax (INS-28), B=1, c=0.15. Gain in a REFERENCE
attention hook (NO sealed-kernel edit) -- validated: hook gamma=1 == triton path (bit-identical first_code,
route all-DCR). Paired per prompt; paired bootstrap 10000 -> 95% CI on Delta_P, on (acc@gamma-acc@1), and on
RECOVERED fraction. SEED=20260531. GPU RTX 4060 Ti.

## WHAT NOT TO DO
Don't change the selected SET in Arm 1 (only the needle's weight). Don't declare TRANSPORT from argmax alone
(continuous Delta_P + step-locus must also be flat). Don't count DEGENERATED as RECOVERED. No causal claim
without Arm1b (+ Arm3 on escalation). Don't tune gamma/CI/bins/locus after results. Reference hook only;
non-destructive; reports to report dir.

## LIMITATIONS
Reference-attention intervention isolates the causal claim (production-kernel application is a separate
engineering question). 1B / semantic K=16 / c=0.15; causal status may be model/task-specific (multi-model arm).
Forcing attention is off-distribution; DEGENERATED at large gamma is expected & is itself an answer. A
COMPUTATION verdict licenses research on flow-redirection memory; it does not make any memory work. Claim is
for THIS regime (retrieval-under-competition), not universal.
