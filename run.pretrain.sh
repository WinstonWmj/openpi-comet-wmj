# set dist training envs
export MASTER_ADDR=${SERVICE_PREFIX}-0.${SUBDOMAIN}
export WORLD_SIZE=${LEPTON_JOB_TOTAL_WORKERS}
export WORLD_RANK=${LEPTON_JOB_WORKER_INDEX}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=12350

config_name=pi05_b1k-pt50_cs32_bs64_lr2.5e-5_step50k_gpu40
exp_name=pi05_b1k-pt50_pretrain

python scripts/compute_norm_stats.py --config-name ${config_name}

python scripts/train_dist.py ${config_name} --exp_name=${exp_name} --overwrite