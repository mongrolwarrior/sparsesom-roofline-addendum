#!/usr/bin/env python3
"""Aggregate the extended ncu captures (ncu2_<impl>_<edge>.csv) into
per-kernel and per-phase roofline tables.

Scaling of the capped edge-256 captures follows scripts/aggregate_phase.py in
sparsesom-paper1: per-tile kernels (csrmm / argmax / rebase) are scaled by
n_tiles / launches_captured; once-per-epoch kernels (norms, fused bmu) are whole.
Counters scale linearly; ratios (intensities, pct-of-peak) are taken from the
captured launches unscaled.

Outputs (in the results dir):
  roofline2_kernels.csv  — per (impl, edge, kernel): time, bytes at DRAM/L2/L1/
                           shared, warp instructions, thread-level op mix, FLOPs,
                           instruction and arithmetic intensities, pipe utilisation
  roofline2_phase.csv    — per (impl, edge): phase totals + intensities
  roofline2_ceilings.csv — device ceilings measured by ncu (peak_sustained x clock)
"""
import glob, math, os, re, sys
import pandas as pd

N_SAMPLES = 26_912_934
SCORES_TILE_BYTES = 512 * 1024 * 1024
PER_TILE = ("csrmm", "argmax", "rebase")
OP = "smsp__sass_thread_inst_executed_op_{}_pred_on.sum"
OPS = ["fadd", "fmul", "ffma", "hadd", "hmul", "hfma", "fp16", "fp32", "conversion",
       "integer", "bit", "control", "memory", "inter_thread_communication", "misc"]
PIPES = ["alu", "fma", "fmaheavy", "fmalite", "lsu", "xu", "cbu"]


def tiles_per_epoch(edge):
    return max(1, math.ceil(N_SAMPLES / (SCORES_TILE_BYTES // (edge * edge * 2))))


def load(path):
    df = pd.read_csv(path, skiprows=2)
    df.columns = [c.strip() for c in df.columns]
    df["val"] = pd.to_numeric(df["Metric Value"].astype(str).str.replace(",", ""), errors="coerce")
    df["k"] = (df["Kernel Name"].str.replace(r"[<(].*", "", regex=True)
               .str.replace(r".*::", "", regex=True).str[:34])
    return df.pivot_table(index=["ID", "k"], columns="Metric Name", values="val", aggfunc="first").reset_index()


def main(results):
    krows, prows, crows = [], [], []
    for path in sorted(glob.glob(os.path.join(results, "ncu2_*_*.csv"))):
        m = re.search(r"ncu2_(.+)_(\d+)\.csv$", os.path.basename(path))
        if not m:
            continue
        impl, edge = m.group(1), int(m.group(2))
        piv = load(path)
        tpe = tiles_per_epoch(edge)

        # ceilings (device constants; identical across launches — take the first)
        r0 = piv.iloc[0]
        sm_hz = r0["sm__cycles_elapsed.avg.per_second"]
        if "dram__bytes.sum.peak_sustained" in piv.columns:
          crows.append({"impl": impl, "edge": edge,
                      "sm_clock_ghz": sm_hz / 1e9,
                      "dram_peak_gbps": r0["dram__bytes.sum.peak_sustained"] * r0["dram__cycles_elapsed.avg.per_second"] / 1e9,
                      "l2_peak_gbps": r0["lts__t_bytes.sum.peak_sustained"] * r0["lts__cycles_elapsed.avg.per_second"] / 1e9,
                      "l1_peak_gbps": r0["l1tex__t_bytes.sum.peak_sustained"] * sm_hz / 1e9,
                      "shared_peak_gbps": r0["sm__sass_data_bytes_mem_shared.sum.peak_sustained"] * sm_hz / 1e9,
                      "warp_inst_peak_gips": r0["smsp__inst_executed.sum.peak_sustained"] * sm_hz / 1e9})

        sums = {"t_s": "gpu__time_duration.sum", "dram_b": None, "l2_b": "lts__t_bytes.sum",
                "l1_b": "l1tex__t_bytes.sum", "shm_b": "sm__sass_data_bytes_mem_shared.sum",
                "warp_inst": "smsp__inst_executed.sum"}
        for k, g in piv.groupby("k"):
            n = g["ID"].nunique()
            scale = tpe / n if any(t in k.lower() for t in PER_TILE) and n < tpe else 1.0
            row = {"impl": impl, "edge": edge, "kernel": k, "launches_captured": n,
                   "scale": round(scale, 2)}
            dram = (g["dram__bytes_read.sum"] + g["dram__bytes_write.sum"]).sum()
            row["time_s"] = g[sums["t_s"]].sum() / 1e9 * scale
            row["dram_tb"] = dram / 1e12 * scale
            row["l2_tb"] = g[sums["l2_b"]].sum() / 1e12 * scale
            row["l1_tb"] = g[sums["l1_b"]].sum() / 1e12 * scale
            row["shared_tb"] = g[sums["shm_b"]].sum() / 1e12 * scale
            wi = g[sums["warp_inst"]].sum() * scale
            row["warp_inst_G"] = wi / 1e9
            for op in OPS:
                row[f"op_{op}_G"] = g[OP.format(op)].sum() * scale / 1e9
            row["flop32_G"] = row["op_fadd_G"] + row["op_fmul_G"] + 2 * row["op_ffma_G"]
            row["flop16_G"] = row["op_hadd_G"] + row["op_hmul_G"] + 2 * row["op_hfma_G"]
            # intensities (scale cancels)
            row["ai_fp32_flop_per_dram_byte"] = row["flop32_G"] * 1e9 / (dram * scale) if dram else None
            row["ai_all_fp_flop_per_dram_byte"] = (row["flop32_G"] + row["flop16_G"]) * 1e9 / (dram * scale) if dram else None
            row["ii_warp_inst_per_dram_sector"] = wi / (dram * scale / 32) if dram else None
            l2b = row["l2_tb"] * 1e12; l1b = row["l1_tb"] * 1e12
            row["ii_warp_inst_per_l2_sector"] = wi / (l2b / 32) if l2b else None
            row["ii_warp_inst_per_l1_sector"] = wi / (l1b / 32) if l1b else None
            row["warp_gips_achieved"] = wi / row["time_s"] / 1e9 if row["time_s"] else None
            row["dram_gbps_achieved"] = dram * scale / row["time_s"] / 1e9 if row["time_s"] else None
            row["dram_pct_peak_mean"] = g["dram__throughput.avg.pct_of_peak_sustained_elapsed"].mean()
            row["sm_pct_peak_mean"] = g["sm__throughput.avg.pct_of_peak_sustained_elapsed"].mean()
            for p in PIPES:
                row[f"pipe_{p}_pct"] = g[f"sm__inst_executed_pipe_{p}.avg.pct_of_peak_sustained_active"].mean()
            krows.append(row)

        kd = pd.DataFrame([r for r in krows if r["impl"] == impl and r["edge"] == edge])
        tot = {"impl": impl, "edge": edge,
               "capture": "exact" if kd["scale"].max() <= 1.0 else f"sampled_x{kd['scale'].max():.0f}",
               "time_s": kd["time_s"].sum(), "dram_tb": kd["dram_tb"].sum(), "l2_tb": kd["l2_tb"].sum(),
               "l1_tb": kd["l1_tb"].sum(), "warp_inst_G": kd["warp_inst_G"].sum(),
               "flop32_G": kd["flop32_G"].sum(), "flop16_G": kd["flop16_G"].sum()}
        for op in OPS:
            tot[f"op_{op}_G"] = kd[f"op_{op}_G"].sum()
        tot["ai_fp32_flop_per_dram_byte"] = tot["flop32_G"] / (tot["dram_tb"] * 1e3)
        tot["ai_all_fp_flop_per_dram_byte"] = (tot["flop32_G"] + tot["flop16_G"]) / (tot["dram_tb"] * 1e3)
        tot["ii_warp_inst_per_dram_sector"] = tot["warp_inst_G"] / (tot["dram_tb"] * 1e3 / 32)
        tot["ii_warp_inst_per_l2_sector"] = tot["warp_inst_G"] / (tot["l2_tb"] * 1e3 / 32)
        tot["ii_warp_inst_per_l1_sector"] = tot["warp_inst_G"] / (tot["l1_tb"] * 1e3 / 32)
        tot["warp_gips_achieved"] = tot["warp_inst_G"] / tot["time_s"]
        tot["dram_gbps_achieved"] = tot["dram_tb"] * 1e3 / tot["time_s"]
        tot["fp32_gflops_achieved"] = tot["flop32_G"] / tot["time_s"]
        prows.append(tot)

    pd.DataFrame(krows).round(4).to_csv(os.path.join(results, "roofline2_kernels.csv"), index=False)
    pd.DataFrame(prows).round(4).to_csv(os.path.join(results, "roofline2_phase.csv"), index=False)
    pd.DataFrame(crows).round(3).drop_duplicates(subset=["sm_clock_ghz", "dram_peak_gbps"]).to_csv(
        os.path.join(results, "roofline2_ceilings.csv"), index=False)
    print(pd.DataFrame(prows)[["impl", "edge", "capture", "time_s", "dram_tb", "warp_inst_G", "flop32_G",
                                "ai_fp32_flop_per_dram_byte", "ii_warp_inst_per_dram_sector",
                                "warp_gips_achieved", "dram_gbps_achieved"]].round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/home/andrew/dev/roofline2-2026-08-24/results")
