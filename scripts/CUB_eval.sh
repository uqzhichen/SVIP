cd ..
python main.py \
--resume checkpoints/CUB_ckpt.pth \
--data_root info-files/ \
--dataset CUB \
--bs 64 \
--epochs 30 \
--attribute_dim 312 \
--wd 0 \
--pre_epochs 3 \
--ce_source 1.0 \
--ce_target 1.0 \
--scale 5 \
--lr 3e-5 \
--pre_lr 5e-5 \
--kl_t 20 \
--kl 1 \
--patch_cls 3 \
--replace_n 1 \
--att_dec 0.3 \
--manual_seed 0 \
--keep_token 40 \
--sim_score attn_rollout \
--schedule_step_size 5 \
--schedule_gamma 0.8 \
--beta 0.5 \
--pretrained_model checkpoints/vit_base_patch16_224.pth \


#80.53 14 79.65 72.80 76.07
