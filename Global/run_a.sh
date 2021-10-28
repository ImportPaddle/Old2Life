python train_domain_A.py --use_v2_degradation --continue_train --training_dataset domain_A \
--name domainA_SR_old_photos --label_nc 0 --loadSize 256 --fineSize 256 \
--dataroot [your_data_folder] \
--no_instance --resize_or_crop crop_only --batchSize 100 --no_html -\
-gpu_ids 0,1,2,3 --self_gen --nThreads 4 --n_downsample_global 3 --k_size 4 --use_v2 --mc 64 --start_r 1 --kl 1 --no_cgan \
--outputs_dir [your_output_folder] \
--checkpoints_dir [your_ckpt_folder]

