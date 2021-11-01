python Global/train_mapping.py \
--use_v2_degradation \
--training_dataset mapping \
--use_vae_which_epoch 200 \
--name mapping_quality \
--label_nc 0 \
--loadSize 256 \
--fineSize 256 \
--no_instance \
--resize_or_crop crop_only \
--batchSize 32 --no_html \
--gpu_ids 0 \
--nThreads 4 \
--l2_feat 60 \
--n_downsample_global 3 \
--mc 64 --k_size 4 \
--start_r 1 --mapping_n_block 6 \
--map_mc 512 --use_l1_feat \
--niter 150 --niter_decay 100 \
--use_vae_which_epoch latest \
--which_opech latest \
--load_pretrainA /home/aistudio/work/Old2Life/test_checkpoints/domainA_SR_old_photos \
--load_pretrainB /home/aistudio/work/Old2Life/test_checkpoints/domainB_old_photos \
--dataroot /home/aistudio/work/Old2Life/test_old \
--outputs_dir /home/aistudio/work/Old2Life/output/ \
--checkpoints_dir /home/aistudio/work/Old2Life/test_checkpoints \
--niter 15 \
--niter_decay 15

