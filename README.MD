# kits


## Commands
Training related commands
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --model InverseVAENetwork --dataset GaussianDataset --name guassian_prediction_low_content --low_loss --checkpoint /mnt/kangfu/Experiments/checkpoints --train_data_root /mnt/kangfu/Experiments/datasets/DIV2K_Flickr2K/ --val_data_root /mnt/kangfu/Experiments/datasets/Set4/ --vgg_loss --gan_mode hinge --batch_size 8 --rkld_loss --lambda_rkld 4 --learning_rate 1e-5 --lambda_gan 2 --lambda_vgg 2 --lambda_content 1 --enorm batch --gnorm batch --sync_bn --resume /mnt/kangfu/Experiments/checkpoints/guassian_prediction_low_contentepoch\=399_v0.ckpt
```