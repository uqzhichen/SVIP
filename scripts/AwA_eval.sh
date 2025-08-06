cd ..
python main.py \
--resume checkpoints/AWA2_ckpt.pth \
--data_root info-files/ \
--dataset AWA2 \
--bs 64 \
--epochs 30 \
--attribute_dim 85 \
--wd 0 \
--pre_epochs 5 \
--ce_source 1.0 \
--ce_target 1.0 \
--scale 5 \
--lr 3e-6 \
--pre_lr 1e-6 \
--kl_t 20 \
--kl 1 \
--patch_cls 4 \
--replace_n 1 \
--att_dec 1.0 \
--manual_seed 0 \
--keep_token 45 \
--sim_score attn_rollout \
--schedule_step_size 5 \
--schedule_gamma 0.8 \
--beta 0.8 \
--pretrained_model checkpoints/vit_base_patch16_224.pth \


 # 68.87 75.54 8  88.25 66.03