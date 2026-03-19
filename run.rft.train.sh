# 单卡训练环境
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 设置缓存和临时目录以避免空间不足问题
# export OPENPI_DATA_HOME="/mnt/project_rlinf_hs/mjwei/repo/openpi-comet/cache/openpi"
# export XDG_CACHE_HOME="/mnt/project_rlinf_hs/mjwei/repo/openpi-comet/cache/xdg"
# export TMPDIR="/mnt/project_rlinf_hs/mjwei/repo/openpi-comet/tmp"
# export JAX_COMPILATION_CACHE_DIR="/mnt/project_rlinf_hs/mjwei/repo/openpi-comet/cache/jax"
# export OMNIGIBSON_DATA_PATH="/mnt/project_rlinf/mjwei/repo/BEHAVIOR-1K/datasets"
# 创建目录
# mkdir -p "$OPENPI_DATA_HOME"
# mkdir -p "$XDG_CACHE_HOME"
# mkdir -p "$TMPDIR"
# mkdir -p "$JAX_COMPILATION_CACHE_DIR"

config_name=pi05_b1k-turning_on_radio_lr2.5e-step50k_rft
exp_name=pi05_b1k-turning_on_radio-rft_pt50_step50k-rollout_v3_bsz256

python scripts/compute_norm_stats.py --config-name ${config_name}

# Resume 训练：设为 true 则从 checkpoint 目录中最后一个 checkpoint 恢复（需去掉 --overwrite，且 exp_name 与之前一致）
RESUME=false

# 单卡 batch=16 时按线性缩放 lr：原 5e-6 * (16/256) ≈ 3e-7
# 或 平方根缩放：new_lr = old_lr × sqrt(new_batch / old_batch)
# 注意：`data.base_config` 在 `DataConfigFactory` 里被 tyro Suppress，不支持用命令行覆盖 `--data.base_config.*`
# Resume 时不要加 --overwrite，checkpoint 目录为 checkpoint_base_dir/exp_name，需与之前训练一致
if [ "$RESUME" = true ]; then
  CMD="python scripts/train.py ${config_name} --exp_name=${exp_name} --resume \
    --weight_loader.params_path=/mnt/project_rlinf_hs/mjwei/download_models/sunshk/comet_submission/pi05-pt50-pretrain-50k/params \
    --batch_size=256 \
    --accumulate_grad_batches=1 \
    --save_interval 1000 \
    --log_interval 10"
else
  CMD="python scripts/train.py ${config_name} --exp_name=${exp_name} --overwrite \
    --weight_loader.params_path=/mnt/project_rlinf_hs/mjwei/download_models/sunshk/comet_submission/pi05-pt50-pretrain-50k/params \
    --batch_size=256 \
    --accumulate_grad_batches=1 \
    --save_interval 1000 \
    --log_interval 10"
fi

LOG_DIR="/mnt/project_rlinf/mjwei/repo/openpi-comet/logs/${exp_name}-$(date +'%Y%m%d-%H:%M:%S')"
mkdir -p "${LOG_DIR}"

MEGA_LOG_FILE="${LOG_DIR}/run.log"
echo "Logging to ${MEGA_LOG_FILE}"

echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
