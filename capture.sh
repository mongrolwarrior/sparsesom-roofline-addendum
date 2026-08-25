#!/usr/bin/env bash
# Instruction-mix roofline capture of the BMU phase — standalone, outside the repos.
# Same six captures / same binaries / same command lines as
# sparsesom-paper1/scripts/ncu_full_epoch.sh (edge 128 exact, edge 256 NCU_CAP=60),
# with an extended metric list (FP/INT/conversion/control SASS op counters, per-pipe
# utilisation, L1/L2/DRAM/shared bytes) for an Instruction Roofline + FP32 roofline.
# Usage: capture.sh EDGE [NCU_CAP]   (IMPLS env: comma list, default all three)
set -euo pipefail
EDGE="$1"; CAP="${2:-}"
OUT=/home/andrew/dev/roofline2-2026-08-24/results
CORPUS=/home/andrew/dev/projects/sparsesom-paper1/data/corpus.train.sbcsr
NCU=/usr/local/cuda-12.8/bin/ncu
SPARSESOM=/home/andrew/dev/projects/SparseBinarySOM/build/sparsesom
STANDARDSPARSESOM=/home/andrew/dev/projects/StandardSparseSOM/build/standardsparsesom
IMPLS="${IMPLS:-ssom-feat,ssom-node,sbsom-bin}"
mkdir -p "$OUT"

M_BASE="dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed"
M_MEM="lts__t_bytes.sum,l1tex__t_bytes.sum,sm__sass_data_bytes_mem_shared.sum"
M_INST="smsp__inst_executed.sum,sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second"
M_OPS="smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__sass_thread_inst_executed_op_hadd_pred_on.sum,smsp__sass_thread_inst_executed_op_hmul_pred_on.sum,smsp__sass_thread_inst_executed_op_hfma_pred_on.sum,smsp__sass_thread_inst_executed_op_fp16_pred_on.sum,smsp__sass_thread_inst_executed_op_fp32_pred_on.sum,smsp__sass_thread_inst_executed_op_conversion_pred_on.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum,smsp__sass_thread_inst_executed_op_bit_pred_on.sum,smsp__sass_thread_inst_executed_op_control_pred_on.sum,smsp__sass_thread_inst_executed_op_memory_pred_on.sum,smsp__sass_thread_inst_executed_op_inter_thread_communication_pred_on.sum,smsp__sass_thread_inst_executed_op_misc_pred_on.sum"
M_PIPE="sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_fmalite.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_cbu.avg.pct_of_peak_sustained_active"
M_PEAK="dram__bytes.sum.peak_sustained,dram__cycles_elapsed.avg.per_second,lts__t_bytes.sum.peak_sustained,lts__cycles_elapsed.avg.per_second,l1tex__t_bytes.sum.peak_sustained,smsp__inst_executed.sum.peak_sustained,sm__sass_data_bytes_mem_shared.sum.peak_sustained"
METRICS="$M_BASE,$M_MEM,$M_INST,$M_OPS,$M_PIPE,$M_PEAK"
CAP_ARG=""; [ -n "$CAP" ] && CAP_ARG="-c $CAP"
SIGMA=$(python3 -c "print(0.5*$EDGE)")

echo "=== [$(date)] roofline2 capture edge $EDGE cap='${CAP:-none}' impls=$IMPLS ==="
case ",$IMPLS," in *,ssom-feat,*)
  echo "--- [$(date)] ssom-feat ---"
  "$NCU" --metrics "$METRICS" $CAP_ARG -k "regex:norms_kernel|csrmm|argmax_kernel|rebase_rowptr" --csv \
    --log-file "$OUT/ncu2_ssom-feat_${EDGE}.csv" \
    "$STANDARDSPARSESOM" "$CORPUS" --map "$EDGE" --layout feature --precision fp16 --stop fixed --epochs 1 --seed 0 2>&1 | tail -2 ;;
esac
case ",$IMPLS," in *,ssom-node,*)
  echo "--- [$(date)] ssom-node ---"
  "$NCU" --metrics "$METRICS" $CAP_ARG -k "regex:norms_kernel|csrmm|argmax_kernel|rebase_rowptr" --csv \
    --log-file "$OUT/ncu2_ssom-node_${EDGE}.csv" \
    "$STANDARDSPARSESOM" "$CORPUS" --map "$EDGE" --layout node --precision fp16 --stop fixed --epochs 1 --seed 0 2>&1 | tail -2 ;;
esac
case ",$IMPLS," in *,sbsom-bin,*)
  echo "--- [$(date)] sbsom-bin ---"
  "$NCU" --metrics "$METRICS" $CAP_ARG -k "regex:norm_fm_kernel|bmu_spmm" --csv \
    --log-file "$OUT/ncu2_sbsom-bin_${EDGE}.csv" \
    "$SPARSESOM" "$CORPUS" --rows "$EDGE" --cols "$EDGE" --bin --epochs 1 --seed 0 --sigma-init "$SIGMA" 2>&1 | tail -2 ;;
esac
echo "=== [$(date)] done edge $EDGE ==="
