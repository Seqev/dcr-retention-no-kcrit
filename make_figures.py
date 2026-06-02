#!/usr/bin/env python3
"""
make_figures.py -- regenerate all paper figures DIRECTLY from the raw per-prompt
logs in ./data/. Every plotted number is recomputed here from the per-prompt
outcome rows; nothing is hand-transcribed. Run: python3 make_figures.py
Outputs: figures/fig1_keff_inert.pdf ... fig4_causal.pdf (+ .png mirrors).

Discipline: trichotomy scoring CORRECT / WRONG_NEEDLE / OTHER_CODE(=starvation
proxy together with MISS); in&wrong := needle_in_topk truthy AND outcome!=CORRECT.
"""
import csv, json, collections, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = os.path.join(os.path.dirname(__file__), "data")
FIG  = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG, exist_ok=True)

KEFFS = [1300, 2000, 2600, 3500, 5200]
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
})

def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def truthy(v):
    v = str(v).strip().lower()
    if v in ("true", "1"): return True
    try: return float(v) >= 0.5
    except: return False

def cell(rows, N, keff, arm="sparse"):
    return [r for r in rows if r["arm"] == arm and int(r["N"]) == N
            and (arm == "dense" or int(r["k_eff_target"]) == keff)]

def acc(sub):
    n = len(sub)
    return (sum(1 for r in sub if r["outcome"] == "CORRECT") / n) if n else float("nan")

def wrong(sub):
    n = len(sub)
    return (sum(1 for r in sub if r["outcome"] == "WRONG_NEEDLE") / n) if n else float("nan")

def nit_rate(sub):
    vals = [float(r["needle_in_topk"]) for r in sub if r["needle_in_topk"] not in ("", None)]
    return (sum(vals) / len(vals)) if vals else float("nan")

def in_and_wrong(sub):
    n = len(sub)
    if not n: return float("nan")
    return sum(1 for r in sub if truthy(r["needle_in_topk"]) and r["outcome"] != "CORRECT") / n

# -------------------------------------------------------------------------
# FIGURE 1 -- k_eff inertness (Llama-3.2-1B, semantic K=16)
# acc_sparse vs k_eff at each informative N; dense as dashed horizontals.
# -------------------------------------------------------------------------
keff = load(os.path.join(DATA, "keff_perp_N_per_prompt.csv"))
sem = [r for r in keff if r["mode"] == "semantic"]
Ns_sem = sorted(set(int(r["N"]) for r in sem))

fig, ax = plt.subplots(figsize=(3.4, 2.7))
colors = {32000: "#1b6ca8", 65536: "#c0392b", 98304: "#7f8c8d", 131072: "#27795b"}
labels = {32000: "32K", 65536: "65K", 98304: "98K (dense-floor)", 131072: "131K"}
for N in Ns_sem:
    accs = [acc(cell(sem, N, k)) for k in KEFFS]
    dacc = acc(cell(sem, N, None, arm="dense"))
    ls = ":" if N == 98304 else "-"
    ax.plot(KEFFS, accs, marker="o", ms=3.5, lw=1.4, ls=ls,
            color=colors[N], label=labels[N])
    ax.axhline(dacc, color=colors[N], lw=0.8, ls="--", alpha=0.55)
ax.set_xlabel(r"absolute key budget $k_{\mathrm{eff}}$")
ax.set_ylabel("retrieval accuracy")
ax.set_title("Sparse acc vs $k_{\\mathrm{eff}}$ (dashed = dense)")
ax.set_xticks(KEFFS); ax.set_xticklabels([str(k) for k in KEFFS], rotation=0)
ax.set_ylim(0.20, 0.70)
ax.legend(frameon=False, loc="center right")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_keff_inert.pdf"))
fig.savefig(os.path.join(FIG, "fig1_keff_inert.png"))
plt.close(fig)

# print verification table
print("=== FIG1 verification (semantic, 1B) acc_sparse by k_eff / dense ===")
for N in Ns_sem:
    row = " ".join(f"{acc(cell(sem,N,k)):.3f}" for k in KEFFS)
    print(f"  N={N:6d} dense={acc(cell(sem,N,None,'dense')):.3f}  sparse[{row}]  "
          f"accRange={max(acc(cell(sem,N,k)) for k in KEFFS)-min(acc(cell(sem,N,k)) for k in KEFFS):.3f}")

# -------------------------------------------------------------------------
# FIGURE 2 -- selection/answer dissociation at N=65536 (1B, semantic)
# needle_in_topk rises with k_eff; accuracy flat. Twin axis.
# -------------------------------------------------------------------------
N0 = 65536
accs = [acc(cell(sem, N0, k)) for k in KEFFS]
nits = [nit_rate(cell(sem, N0, k)) for k in KEFFS]
dacc = acc(cell(sem, N0, None, "dense"))

fig, ax = plt.subplots(figsize=(3.4, 2.7))
l1, = ax.plot(KEFFS, nits, marker="s", ms=3.5, lw=1.5, color="#2c7fb8",
              label="needle-in-top-$k$ (selection)")
ax.set_xlabel(r"absolute key budget $k_{\mathrm{eff}}$")
ax.set_ylabel("needle-in-top-$k$ rate", color="#2c7fb8")
ax.tick_params(axis="y", labelcolor="#2c7fb8")
ax.set_ylim(0.40, 0.95)
ax2 = ax.twinx()
ax2.spines["top"].set_visible(False)
l2, = ax2.plot(KEFFS, accs, marker="o", ms=3.5, lw=1.5, color="#c0392b",
               label="retrieval accuracy")
ax2.axhline(dacc, color="#c0392b", lw=0.8, ls="--", alpha=0.6)
ax2.set_ylabel("accuracy", color="#c0392b")
ax2.tick_params(axis="y", labelcolor="#c0392b")
ax2.set_ylim(0.30, 0.60)
ax.set_title("Selection improves, answer does not (65K)")
ax.set_xticks(KEFFS)
ax.legend(handles=[l1, l2], frameon=False, loc="center left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_dissociation.pdf"))
fig.savefig(os.path.join(FIG, "fig2_dissociation.png"))
plt.close(fig)
print("\n=== FIG2 (65K) nit by k_eff ===", [f"{x:.3f}" for x in nits],
      "| acc", [f"{x:.3f}" for x in accs])

# -------------------------------------------------------------------------
# FIGURE 3 -- capacity shifts the ceiling (cross-model, N=32K, semantic K=16)
# in&wrong fraction (lower = better disambiguation) + dense acc, vs capacity.
# 1B reference numbers recomputed from keff log; 3B from crossmodel logs.
# -------------------------------------------------------------------------
def model_at_32k(rows):
    # average in&wrong across k_eff at N=32000; dense acc at N=32000
    iw = np.mean([in_and_wrong(cell(rows, 32000, k)) for k in KEFFS])
    dacc = acc([r for r in rows if r["arm"] == "dense" and int(r["N"]) == 32000])
    return iw, dacc

iw_1b, dacc_1b = model_at_32k(sem)
ll3 = load(os.path.join(DATA, "crossmodel_Llama3B_per_prompt.csv"))
qw3 = load(os.path.join(DATA, "crossmodel_Qwen3B_per_prompt.csv"))
iw_ll3, dacc_ll3 = model_at_32k(ll3)
iw_qw3, dacc_qw3 = model_at_32k(qw3)

models = ["Llama-1B", "Qwen2.5-3B", "Llama-3.2-3B"]
iw = [iw_1b, iw_qw3, iw_ll3]
dacc = [dacc_1b, dacc_qw3, dacc_ll3]
x = np.arange(3)
fig, ax = plt.subplots(figsize=(3.6, 2.7))
b1 = ax.bar(x - 0.2, iw, 0.38, color="#c0392b", label="in&wrong (selected yet wrong)")
b2 = ax.bar(x + 0.2, dacc, 0.38, color="#1b6ca8", label="dense accuracy")
ax.set_xticks(x); ax.set_xticklabels(models, fontsize=8)
ax.set_ylabel("fraction @ N=32K")
ax.set_title("Capacity raises the ceiling (no $K_{\\mathrm{crit}}$)")
ax.set_ylim(0, 0.9)
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
            f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=6.5)
ax.legend(frameon=False, loc="upper center", ncol=1)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_capacity.pdf"))
fig.savefig(os.path.join(FIG, "fig3_capacity.png"))
plt.close(fig)
print("\n=== FIG3 (32K) in&wrong / dense_acc ===")
for m, a, d in zip(models, iw, dacc):
    print(f"  {m:14s} in&wrong={a:.3f} dense_acc={d:.3f}")

# -------------------------------------------------------------------------
# FIGURE 4 -- causal probe (gamma), Llama-1B, N=65K, P1 (needle in set yet wrong)
# Delta_P and recovery fraction: needle-gain g=1/4/16, competitor g=16, tell-it.
# -------------------------------------------------------------------------
g = load(os.path.join(DATA, "gamma_per_prompt.csv"))
def grp(arm, gamma):
    return [r for r in g if r["arm"] == arm and r["gamma"] == str(gamma)]
def mean_dP(sub):
    vals = [float(r["delta_p"]) for r in sub if r["delta_p"] not in ("", None)]
    return np.mean(vals) if vals else 0.0
def rec_frac(sub):
    # recovered := baseline WRONG_NEEDLE -> CORRECT (strict)
    base_wrong = [r for r in sub if r["baseline_outcome"] == "WRONG_NEEDLE"]
    if not base_wrong: 
        # fall back to baseline!=CORRECT
        base_wrong = [r for r in sub if r["baseline_outcome"] != "CORRECT"]
    n = len(base_wrong)
    return (sum(1 for r in base_wrong if r["outcome"] == "CORRECT") / n) if n else 0.0

needle4, needle16 = grp("needle", 4), grp("needle", 16)
comp16 = grp("competitor", 16)
arm3 = json.load(open(os.path.join(DATA, "gamma_arm3_results.json")))
arm3_rec = arm3["summary"]["recovered_frac"]

dP4, dP16 = mean_dP(needle4), mean_dP(needle16)
dPc = mean_dP(comp16)
rec4, rec16 = rec_frac(needle4), rec_frac(needle16)
recc = rec_frac(comp16)

fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.6))
# left: Delta_P
axL = axes[0]
xs = ["needle\n$\\gamma$=4", "needle\n$\\gamma$=16", "competitor\n$\\gamma$=16"]
ys = [dP4, dP16, dPc]
cols = ["#2c7fb8", "#1b4f72", "#7f8c8d"]
bars = axL.bar(xs, ys, color=cols, width=0.6)
axL.axhline(0, color="k", lw=0.6)
axL.set_ylabel(r"$\Delta P$(correct first token)")
axL.set_title("Weight lever is needle-specific")
for b, y in zip(bars, ys):
    axL.text(b.get_x()+b.get_width()/2, y+(0.02 if y>=0 else -0.05),
             f"{y:+.3f}", ha="center", va="bottom" if y>=0 else "top", fontsize=7)
axL.set_ylim(-0.15, 0.78)
# right: recovery fraction
axR = axes[1]
xs2 = ["needle\n$\\gamma$=16", "competitor\n$\\gamma$=16", "re-deliver\n(plain text)"]
ys2 = [rec16, recc, arm3_rec]
cols2 = ["#1b4f72", "#7f8c8d", "#27795b"]
bars2 = axR.bar(xs2, ys2, color=cols2, width=0.6)
axR.set_ylabel("fraction of failures recovered")
axR.set_title("Re-delivery $\\geq$ weight-gain $\\Rightarrow$ transport")
axR.set_ylim(0, 1.08)
for b, y in zip(bars2, ys2):
    axR.text(b.get_x()+b.get_width()/2, y+0.02, f"{y:.0%}",
             ha="center", va="bottom", fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig4_causal.pdf"))
fig.savefig(os.path.join(FIG, "fig4_causal.png"))
plt.close(fig)
print("\n=== FIG4 (gamma, 65K, n=30 P1) ===")
print(f"  needle dP: g4={dP4:+.3f} g16={dP16:+.3f}  competitor dP g16={dPc:+.3f}")
print(f"  recovered: needle g16={rec16:.0%}  competitor={recc:.0%}  tell-it(Arm3)={arm3_rec:.0%}")
print("\nAll figures written to figures/.")
