#!/usr/bin/env python3
"""Two rooflines from roofline2_kernels.csv + roofline2_ceilings.csv (RTX 4090, ncu-measured).

Fig A — Instruction Roofline (Ding & Williams 2019): y = warp instructions/s, x = warp
        instructions per 32-byte sector; three bandwidth slopes (DRAM, L2, L1) + flat issue roof.
        Each kernel is drawn as three connected points (its intensity at L1, L2, DRAM).
Fig B — Conventional FP32 roofline: y = FP32 GFLOP/s, x = FLOP per DRAM byte; FP32 peak roof,
        DRAM slope, L2 slope.
Outputs PNG + PDF + the plotted point table (roofline2_points.csv).
"""
import sys, os
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = sys.argv[1] if len(sys.argv) > 1 else "/home/andrew/dev/roofline2-2026-08-24/results"
k = pd.read_csv(os.path.join(R, "roofline2_kernels.csv"))
c = pd.read_csv(os.path.join(R, "roofline2_ceilings.csv")).iloc[0]

# device ceilings (measured by ncu; FP32 peak from SM count x lanes x 2 x measured clock)
SM_CLK = c.sm_clock_ghz * 1e9
DRAM, L2, L1 = c.dram_peak_gbps * 1e9, c.l2_peak_gbps * 1e9, c.l1_peak_gbps * 1e9   # B/s
ISSUE = c.warp_inst_peak_gips * 1e9                                                  # warp-inst/s
FP32_PEAK = 128 * 128 * 2 * SM_CLK                                                   # FLOP/s (4090: 128 SMs)

SERIES = [  # (label, selector, colour, marker) — fixed order; palette validated (dataviz)
    ("sbsom-bin fused BMU (bmu_spmm)", lambda d: (d.impl == "sbsom-bin") & d.kernel.str.contains("bmu_spmm"), "#2a78d6", "o"),
    ("cuSPARSE csrmm, feature-major",   lambda d: (d.impl == "ssom-feat") & d.kernel.str.contains("csrmm"), "#eb6834", "s"),
    ("cuSPARSE csrmm, node-major",      lambda d: (d.impl == "ssom-node") & d.kernel.str.contains("csrmm"), "#1baf7a", "D"),
    ("argmax read-back (both ssom)",    lambda d: (d.impl == "ssom-feat") & d.kernel.str.contains("argmax"), "#eda100", "^"),
]
INK, INK2, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.titleweight": "bold"})

points = []


def roof(ax, slope_bps, flat, label, x, unit=32.0):
    """Draw min(slope*x, flat) for a bandwidth ceiling expressed per sector (unit bytes)."""
    y = np.minimum(slope_bps / unit * x, flat) / 1e9          # axes are in G
    ax.plot(x, y, color=INK2, lw=1, ls="--")
    xi = x[np.argmin(np.abs(y - flat / 1e9 * 0.25))]           # label a quarter of the way up the slope
    ax.annotate(label, (xi, slope_bps / unit * xi / 1e9), fontsize=7.5, color=INK2, rotation=0,
                xytext=(4, -10), textcoords="offset points")


# ── Fig A: Instruction Roofline ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
x = np.logspace(-1.5, 2.5, 300)
for ax, edge in zip(axes, (128, 256)):
    for slope, name in ((DRAM, "DRAM"), (L2, "L2"), (L1, "L1")):
        roof(ax, slope, ISSUE, f"{name} {slope/1e12:.1f} TB/s", x)
    ax.axhline(ISSUE / 1e9, color=INK2, lw=1)
    ax.text(x[-1], ISSUE / 1e9 * 1.08, f"warp-issue peak {ISSUE/1e9:.0f} G inst/s", ha="right", fontsize=7.5, color=INK2)
    for label, sel, col, mk in SERIES:
        d = k[sel(k) & (k.edge == edge)]
        if d.empty:
            continue
        r = d.iloc[0]
        xs = [r.ii_warp_inst_per_l1_sector, r.ii_warp_inst_per_l2_sector, r.ii_warp_inst_per_dram_sector]
        ys = [r.warp_gips_achieved] * 3
        ax.plot(xs, ys, color=col, lw=1, alpha=0.6)
        ax.scatter(xs, ys, s=[28, 28, 64], color=col, marker=mk, edgecolor="white", linewidth=1, zorder=3,
                   label=label if edge == 128 else None)
        ax.annotate(f"{r.warp_gips_achieved:.0f}", (xs[2], ys[2]), xytext=(6, -3), textcoords="offset points",
                    fontsize=7.5, color=INK)
        for lvl, xv in zip(("L1", "L2", "DRAM"), xs):
            points.append({"figure": "instruction", "edge": edge, "series": label, "level": lvl,
                           "x_warp_inst_per_sector": xv, "y_G_warp_inst_per_s": r.warp_gips_achieved})
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(x[0], x[-1]); ax.set_ylim(5, 3000)
    ax.set_xlabel("warp instructions per 32-B sector  (points: L1 · L2 · DRAM, left→right)")
    ax.set_title(f"Instruction roofline — BMU phase, {edge}×{edge}")
    ax.grid(True, which="major", color=GRID, lw=0.6); ax.set_axisbelow(True)
axes[0].set_ylabel("warp instructions / s  (G)")
axes[0].legend(loc="lower right", frameon=False, fontsize=7.5)
fig.suptitle("RTX 4090, one epoch, fp16 codebooks; ceilings measured by Nsight Compute", fontsize=9, color=INK2, y=0.995)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(R, f"fig_instruction_roofline.{ext}"), dpi=200)
plt.close(fig)

# ── Fig B: Conventional FP32 roofline ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.4, 4.6))
x = np.logspace(-1, 3, 300)
for slope, name in ((DRAM, "DRAM"), (L2, "L2")):
    roof(ax, slope, FP32_PEAK, f"{name} {slope/1e12:.1f} TB/s", x, unit=1.0)
ax.axhline(FP32_PEAK / 1e9, color=INK2, lw=1)
ax.text(x[-1], FP32_PEAK / 1e9 * 1.1, f"FP32 peak {FP32_PEAK/1e12:.1f} TFLOP/s", ha="right", fontsize=7.5, color=INK2)
ridge = FP32_PEAK / DRAM
ax.axvline(ridge, color=INK2, lw=0.6, ls=":"); ax.text(ridge * 1.15, 2.5e4, f"DRAM ridge\n{ridge:.0f} FLOP/B", fontsize=7.5, color=INK2)
for label, sel, col, mk in SERIES:
    for edge, size in ((128, 40), (256, 80)):
        d = k[sel(k) & (k.edge == edge)]
        if d.empty:
            continue
        r = d.iloc[0]
        gf = r.flop32_G / r.time_s
        ax.scatter(r.ai_fp32_flop_per_dram_byte, gf, s=size, color=col, marker=mk, edgecolor="white",
                   linewidth=1, zorder=3, label=f"{label}" if edge == 128 else None)
        ax.annotate(f"{edge}²", (r.ai_fp32_flop_per_dram_byte, gf),
                    xytext=(7, 4) if edge == 128 else (-9, -12), textcoords="offset points",
                    ha="left" if edge == 128 else "right", fontsize=7.5, color=INK)
        points.append({"figure": "conventional", "edge": edge, "series": label, "level": "DRAM",
                       "x_flop_per_dram_byte": r.ai_fp32_flop_per_dram_byte, "y_GFLOP_per_s": gf,
                       "pct_of_fp32_peak": 100 * gf * 1e9 / FP32_PEAK})
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(x[0], x[-1]); ax.set_ylim(5, 2e5)
ax.set_xlabel("FP32 FLOP per DRAM byte"); ax.set_ylabel("FP32 GFLOP / s")
ax.set_title("Conventional roofline — BMU-phase kernels")
ax.grid(True, which="major", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(loc="lower right", bbox_to_anchor=(0.99, 0.04), frameon=False, fontsize=7.5)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(R, f"fig_conventional_roofline.{ext}"), dpi=200)
plt.close(fig)

pd.DataFrame(points).round(4).to_csv(os.path.join(R, "roofline2_points.csv"), index=False)
print("ceilings: DRAM %.0f GB/s  L2 %.0f GB/s  L1 %.0f TB/s  issue %.0f G inst/s  FP32 %.1f TFLOP/s  ridge %.0f FLOP/B"
      % (DRAM / 1e9, L2 / 1e9, L1 / 1e12, ISSUE / 1e9, FP32_PEAK / 1e12, ridge))
