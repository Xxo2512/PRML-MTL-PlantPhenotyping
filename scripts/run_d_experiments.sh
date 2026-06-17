#!/usr/bin/env bash
# 服务器一键跑: D 部分 TaskPrompter 实验
#
# 用法 (在服务器上, conda 环境, cwd=项目根):
#   bash scripts/run_d_experiments.sh
#
# 跑完产物:
#   checkpoints/single_<task>_5ep/last.pt          × 4   (单任务 baseline)
#   checkpoints/taskprompter_20ep_ps/last.pt              (TaskPrompter 全量)
#   logs/results.csv                                      (追加 5 行评测)
#
# 时间预估 (RTX 4090, swin_tiny, bs_per_task=8, 384):
#   single_seg          : ~5  min/epoch ×  5 ep ≈ 25 min
#   single_det          : ~10 min/epoch ×  5 ep ≈ 50 min
#   single_cnt          : ~5  min/epoch ×  5 ep ≈ 25 min
#   single_cls          : ~30 min/epoch ×  5 ep ≈ 2.5 h
#   TaskPrompter 全量    : ~25 min/epoch × 20 ep ≈ 8 h     (12.9M 可训参数, 比 MTLoRA 轻 25%)
#   合计 ≈ 12 h, 4090 单卡, 一晚跑完
set -euo pipefail
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export PYTHONUNBUFFERED=1

VANILLA_CFG=configs/method/vanilla.yaml
TASKPROMPTER_CFG=configs/method/taskprompter.yaml

mkdir -p logs

run_train_eval () {
    local cfg=$1; local tag=$2; shift 2
    echo "==================== train: $tag ===================="
    python train.py --config "$cfg" --tag "$tag" "$@"
    echo "==================== eval : $tag ===================="
    python evaluate.py --config "$cfg" --tag "$tag" \
                       --ckpt "checkpoints/$tag/last.pt" || true
}

# ═══════════════════════════════════════════════════════════════
# 1) 单任务 baseline (4 个), 各 5 epoch
#    使用 vanilla config + --single_task, 作为 diagonal 对照
# ═══════════════════════════════════════════════════════════════
for t in seg det cnt cls; do
    run_train_eval "$VANILLA_CFG" "single_${t}_5ep" --epochs 5 --single_task "$t"
done

# ═══════════════════════════════════════════════════════════════
# 2) TaskPrompter 全量 MTL, 20 epoch (PS 调度)
# ═══════════════════════════════════════════════════════════════
run_train_eval "$TASKPROMPTER_CFG" "taskprompter_20ep_ps" --epochs 20

echo
echo "[ALL DONE] check logs/results.csv"
