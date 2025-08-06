cd ..
python main.py \
--data_root info-files/ \
--dataset SUN \
--bs 64 \
--epochs 50 \
--attribute_dim 102 \
--wd 0 \
--pre_epochs 3 \
--ce_source 1.0 \
--ce_target 1.0 \
--scale 5 \
--lr 1e-5 \
--pre_lr 1e-4 \
--kl_t 20 \
--kl 1 \
--patch_cls 3 \
--replace_n 1 \
--att_dec 0.3 \
--num_workers 8 \
--manual_seed 0 \
--keep_token 45 \
--sim_score attn_rollout \
--schedule_step_size 5 \
--schedule_gamma 0.8 \
--beta 0.5 \
--pretrained_model checkpoints/vit_base_patch16_224.pth \

# 72.36 seen=48.49% unseen=54.17%, H=51.17%