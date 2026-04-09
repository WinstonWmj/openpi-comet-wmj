# 单卡训练环境
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 设置缓存和临时目录以避免空间不足问题（根分区只有100G，必须重定向到NFS大盘）
export OPENPI_DATA_HOME="/mnt/project_rlinf_hs/mjwei/repo/openpi-comet/cache/openpi"
export XDG_CACHE_HOME="/mnt/project_rlinf_hs/mjwei/repo/openpi-comet/cache/xdg"
export HF_HOME="/mnt/project_rlinf_hs/mjwei/repo/openpi-comet/cache/huggingface"
export TMPDIR="/mnt/project_rlinf_hs/mjwei/repo/openpi-comet/tmp"
export JAX_COMPILATION_CACHE_DIR="/mnt/project_rlinf_hs/mjwei/repo/openpi-comet/cache/jax"
# export OMNIGIBSON_DATA_PATH="/mnt/project_rlinf/mjwei/repo/BEHAVIOR-1K/datasets"
mkdir -p "$OPENPI_DATA_HOME" "$XDG_CACHE_HOME" "$HF_HOME" "$TMPDIR" "$JAX_COMPILATION_CACHE_DIR"

config_name=pi05_b1k-pickplace-lr2.5e-step20k-200-pt50
exp_name=pi05_b1k-pickplace-lr2.5e-step20k-200-pt50

# python scripts/compute_norm_stats.py --config-name ${config_name}

# 单卡 batch=16 时按线性缩放 lr：原 5e-6 * (16/256) ≈ 3e-7
# 或 平方根缩放：new_lr = old_lr × sqrt(new_batch / old_batch)
# 注意：`data.base_config` 在 `DataConfigFactory` 里被 tyro Suppress，不支持用命令行覆盖 `--data.base_config.*`
CMD="python scripts/train.py ${config_name} --exp_name=${exp_name} --overwrite \
  --weight_loader.params_path=/mnt/project_rlinf_hs/mjwei/download_models/sunshk/comet_submission/pi05-pt50-pretrain-50k/params \
  --batch_size=256 \
  --save_interval 5000 \
  --log_interval 10"

LOG_DIR="/mnt/project_rlinf/mjwei/repo/openpi-comet/logs/${exp_name}-$(date +'%Y%m%d-%H:%M:%S')"
mkdir -p "${LOG_DIR}"

MEGA_LOG_FILE="${LOG_DIR}/run.log"
echo "Logging to ${MEGA_LOG_FILE}"

echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
