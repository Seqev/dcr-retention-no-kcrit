# No Critical Key Budget in Query-Axis Top-K Sparse Attention

**Long-context retention is bounded by the base model, not the compressor.**

Evgenii Vyaltsev (ORCID 0009-0004-3712-6798), Daniil Vyaltsev — June 2026.


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20514229.svg)](https://doi.org/10.5281/zenodo.20514229)


This repository accompanies the preprint of the same name. It contains the paper
source, the compiled PDF, the **raw per-prompt logs** for every experiment, and a
single script that **recomputes all figures directly from those logs**.

---

## TL;DR

Top-K sparse attention is usually justified by the implicit belief that there is a
*critical key budget* below which long-context ability collapses. We test that
directly with a training-free, **exact-ranking** query-axis selector
(top-K by ⟨u_Q, K_i⟩, u_Q = Q/‖Q‖) and find:

1. **The key budget is inert.** A 4× swing of the absolute budget `k_eff` over
   [1300, 5200] moves retrieval accuracy by ≤ 0.03 at any context length `N`, on
   three models. There is **no absolute `K_crit`** — the same budget `k_eff=5200`
   holds at 32K yet fails at 65K.
2. **The limit is the base model.** Dense (full) attention degrades the same way as
   sparse with growing `N`; the needle is *selected* into the top-k ever more often
   as the budget grows, yet the answer does not improve. The bottleneck is
   **disambiguation in the weights**, not selection.
3. **Capacity raises the ceiling, not a budget threshold.** On Qwen2.5-3B and
   Llama-3.2-3B (kernel port validated *bitwise-exact* vs a pure-PyTorch reference),
   `K_crit` still does not emerge; instead the selected-yet-wrong fraction at N=32K
   falls 0.37 → 0.23 → 0.18 with capacity. Verdict: **no-K_crit /
   weights-bound-universal / capacity-shifts-ceiling**.
4. **Attention here is transport, not computation.** Up-weighting the needle's
   attention within a fixed selected set recovers most failures, but plain
   re-delivery of the same fact recovers at least as much (100% ≥ 80%) — so the
   correctness bottleneck is downstream of attention.

**Practical takeaway:** within the practical operating regime there is *no
efficiency cliff in the compressor*. The actionable lever for "smarter" on a fixed
model is **capacity or better evidence delivery**, not the sparsity budget; sparsity
buys speed and length at iso-ability. Safe operating coverage `c ≳ 0.10` to 128K.

---

## Repository layout

```
paper3_retention.tex        LaTeX source (arXiv-ready)
paper3_retention.pdf        compiled preprint (9 pp)
make_figures.py             regenerates all figures from data/ (no hand-typed numbers)
figures/                    fig1..fig4 (.pdf + .png), produced by make_figures.py
data/
  keff_perp_N_per_prompt.csv        decoupling experiment, Llama-3.2-1B (per-prompt)
  keff_perp_N_results.jsonl         per-cell aggregates
  crossmodel_Llama3B_per_prompt.csv cross-model Gate B, Llama-3.2-3B
  crossmodel_Qwen3B_per_prompt.csv  cross-model Gate B, Qwen2.5-3B
  gamma_per_prompt.csv              causal probe, Llama-3.2-1B
  gamma_arm3_results.json           causal probe, re-delivery control
PREREGISTRATIONS/           the pre-registration for each gate (fixed before results)
CITATION.cff                citation metadata
```

## Reproduce the figures

```bash
python3 -m pip install matplotlib numpy
python3 make_figures.py        # writes figures/*.pdf and *.png; prints a verification table
```

`make_figures.py` reads only the per-prompt logs in `data/` and recomputes every
plotted quantity (accuracy, needle-in-top-k rate, in&wrong fraction, ΔP, recovery
fractions). Nothing is transcribed by hand.

## Rebuild the PDF

```bash
pdflatex paper3_retention.tex && pdflatex paper3_retention.tex
```

---

## Method notes (for reviewers)

- **Selector is exact.** Ranking by ⟨u_Q, K_i⟩ is order-identical to ranking by
  ⟨Q, K_i⟩; the Triton kernel is validated equal to a pure-PyTorch reference, and
  the cross-model port is `max|ΔO| = 0.0`, index agreement `1.0` at GQA ratios
  4/8/3.
- **Matched dense control.** `enable_dcr=False` (all-SDPA) is decoded once per
  prompt; sparse is decoded at each `k_eff` on the *same* prompts (paired). Route
  counters checked per cell; a silent SDPA fallback invalidates the cell.
- **Three-way scoring.** CORRECT / WRONG-NEEDLE (mis-selection) / STARVATION
  (other-code or miss). WRONG-NEEDLE is never collapsed into miss.
- **Pre-registration.** Grid, thresholds, monotone rule and verdict map are fixed
  before each run (see `PREREGISTRATIONS/`); none is tunable after.
- **Scope honesty.** Greedy, no chain-of-thought; semantic / near-duplicate
  retrieval only; multi-hop synthesis floors on these bases and is *deferred*.
  Any claim about the exact coverage onset c* at 128K is single-seed and provisional;
  the headline (budget-inert, no-K_crit, weights-bound) does not depend on it.

## Status

Preprint v1 for priority. A multi-seed robustness repeat of the 128K boundary cells
and a synthesis (multi-hop) gate on a base that does not floor are the immediate
follow-ups; neither changes the headline null.

## License

- Paper text, figures: CC-BY-4.0.
- Code (`make_figures.py`): MIT.
- Data (`data/`): CC-BY-4.0.

## Citation

**DOI:** [10.5281/zenodo.20514229](https://doi.org/10.5281/zenodo.20514229)

> Vyaltsev, E., & Vyaltsev, D. (2026). *No Critical Key Budget in Query-Axis Top-K
> Sparse Attention: Long-Context Retention Is Bounded by the Base Model, Not the
> Compressor.* Zenodo. https://doi.org/10.5281/zenodo.20514229

```bibtex
@misc{vyaltsev2026nokcrit,
  author       = {Vyaltsev, Evgenii and Vyaltsev, Daniil},
  title        = {No Critical Key Budget in Query-Axis Top-K Sparse Attention:
                  Long-Context Retention Is Bounded by the Base Model, Not the Compressor},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20514229},
  url          = {https://doi.org/10.5281/zenodo.20514229}
}
```

The DOI above is the version DOI for v1.0.0. The concept DOI (always resolving to the
latest version) is shown on the Zenodo record page.
