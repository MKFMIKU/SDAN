# kits


## Commands
```bash
# For DFS Baseline
python main.py --model DFSNetwork --dataset SRRawDataset --name DFS_baseline --checkpoint /data/Experiments/DFS/baseline --train_data_root /tmp_data/running/SRRaw/train/ --val_data_root /tmp_data/running/SRRaw/test/ --batch_size 1 --workers 4 --val 10 --learning_rate 1e-3

# For Ours
CUDA_VISIBLE_DEVICES=0 python main.py --model DFSNetwork --dataset SRRawDataset --name DFS_baseline --checkpoint ./ck --train_data_root /data/datasets/SRRAW/train --val_data_root /data/datasets/SRRAW/test  --batch_size 8 --val 10 --learning_rate 2e-4 --sdcn --n_feats 64 --workers 16

# For DFS test
python main.py --model DFSNetwork --dataset SRRawDataset --name sdcn_color_long --checkpoint /data/Experiments/DFS/sdcn_color_long/ --train_data_root /tmp_data/running/SRRaw/train/ --val_data_root /tmp_data/running/SRRaw/test/ --batch_size 1 --workers 4 --val 10 --learning_rate 1e-3 --sdcn --resume /data/Experiments/DFS/sdcn_color_nocpa-sdcn_color_nocpa-v29.ckpt --test --save_output /data/Experiments/DFS/sdcn_color_619_4x_nocpa_results/
```


## Notices
- For the SR-Raw dataset, you should install `rawpy==0.14.0` and `pillow==7.1.2`
