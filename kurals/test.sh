CUDA_VISIBLE_DEVICES=4 python test.py \
    --cfg config_files/kuralsnet.json \
    --mode prec \
    --dataset KuRALS_CW \
    --model-path /media/data2/liteng/Project/Radar/KuRALS/test_results/kuralsnet_cw.pt \
    >test_kuralsnet_kuralscw.log 2>&1