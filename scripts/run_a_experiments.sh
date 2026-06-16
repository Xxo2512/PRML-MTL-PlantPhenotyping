#!/usr/bin/env bash
# 服务器一键跑: A 部分的 W13/W14 实验
#
# 用法 (假设你在 conda 环境里且 cwd=项目根):
#   bash scripts/run_a_experiments.sh
#
# 跑完产物:
#   checkpoints/single_<task>_<exp>/last.pt    × 4   (单任务 baseline)
#   checkpoints/vanilla_swint_384_rr/last.pt          (vanilla MTL)
#   checkpoints/mtlora_swint_384_rr/last.pt           (MTLoRA)
#   logs/results.csv                                  (8 行评测)
#
# 时间预估 (RTX 4090, swin_tiny, bs_per_task=8, 384):
#   single_seg  : ~5  min/epoch ×  5 ep ≈ 25 min
#   single_det  : ~10 min/epoch ×  5 ep ≈ 50 min
#   single_cnt  : ~5  min/epoch ×  5 ep ≈ 25 min
#   single_cls  : ~30 min/epoch ×  5 ep ≈ 2.5 h
#   vanilla MTL : ~30 min/epoch × 10 ep ≈ 5 h
#   MTLoRA      : ~30 min/epoch × 20 ep ≈ 10 h     (LoRA 训练需要更多 epoch)
#   合计 ~18 h, 4090 单卡, 一晚跑完
set -euo pipefail
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export PYTHONUNBUFFERED=1

VANILLA_CFG=configs/method/vanilla.yaml
MTLORA_CFG=configs/method/mtlora.yaml

mkdir -p logs

run_train_eval () {
    local cfg=$1; local tag=$2; shift 2
    echo "==================== train: $tag ===================="
    python train.py --config "$cfg" --tag "$tag" "$@"
    echo "==================== eval : $tag ===================="
    python evaluate.py --config "$cfg" --tag "$tag" \
                       --ckpt "checkpoints/$tag/last.pt" || true
}

# 1) single-task baseline (4 个), 各 5 epoch
for t in seg det cnt cls; do
    run_train_eval "$VANILLA_CFG" "single_${t}_5ep" --epochs 5 --single_task "$t"
done

# 2) vanilla MTL, 10 epoch
run_train_eval "$VANILLA_CFG" "vanilla_10ep_ps" --epochs 10

# 3) MTLoRA, 20 epoch (LoRA 需更多)
run_train_eval "$MTLORA_CFG"  "mtlora_20ep_ps"  --epochs 20

echo
echo "[ALL DONE] check logs/results.csv"
