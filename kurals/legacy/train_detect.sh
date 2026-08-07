cd "$(dirname "$0")"

CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch \
    --master_port 8121 \
    --nproc_per_node=1 \
    --use_env train_detect.py \
    --cfg kuralsnet_npu.json \
    --dataset KuRALS_CW \
    >train_kuralsnet_npu_kuralscw.log 2>&1
