CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch \
    --master_port 8120 \
    --nproc_per_node=1 \
    --use_env train.py \
    --cfg config_files/kuralsnet.json \
    --dataset KuRALS_CW \
    >train_kuralsnet_kuralscw.log 2>&1