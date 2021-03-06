python Global/train_domain_A.py \
--use_v2_degradation \
--training_dataset domain_A \
--name domainA_SR_old_photos \
--label_nc 0 \
--loadSize 256 \
--fineSize 256 \
--no_instance \
--resize_or_crop crop_only \
--batchSize 64 --no_html \
--gpu_ids 0 \
--self_gen \
--nThreads 4 \
--n_downsample_global 3 \
--use_vae_which_epoch latest \
--which_epoch latest \
--k_size 4 --use_v2 --mc 64 --start_r 1 --kl 1 --no_cgan \
--dataroot D:\\Desktop\\plan\\Old2Life\\Global\\test_old \
--outputs_dir /home/aistudio/work/Old2Life/output/ \
--checkpoints_dir /home/aistudio/work/Old2Life/checkpoints \
--niter 15 \
--niter_decay 15 \
--continue_train
