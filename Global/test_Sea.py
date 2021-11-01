# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.mapping_model import Pix2PixHDModel_Mapping, InferenceModel
import util.util as util
from util.visualizer import Visualizer
import os
import numpy as np

import paddle
import torchvision_paddle.utils as vutils
import datetime
import paddle.distributed as dist
import random

opt = TrainOptions().parse()
visualizer = Visualizer(opt)

opt.display_freq = 1
opt.print_freq = 1
opt.niter = 1
opt.niter_decay = 0
opt.max_dataset_size = 10

start_epoch, epoch_iter = 0, 0

opt.start_epoch = start_epoch
### temp for continue train unfixed decoder

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(dataset) * opt.batchSize
print('#training images = %d' % dataset_size)

model = InferenceModel()
model.initialize(opt)

path = os.path.join(opt.checkpoints_dir, opt.name, 'model.txt')
fd = open(path, 'w')

if opt.use_skip_model:
    fd.write(str(model.mapping_net))
    fd.close()
else:
    fd.write(str(model.netG_A))
    fd.write(str(model.mapping_net))
    fd.close()

total_steps = (start_epoch - 1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq
### used for recovering training

performance = util.compute_performance()
performance_all = util.compute_performance()

epoch_s_t = datetime.datetime.now()

model.eval()
for i, data in enumerate(dataset, start=epoch_iter):
    iter_start_time = time.time()
    total_steps += opt.batchSize
    epoch_iter += opt.batchSize

    # whether to collect output images
    save_fake = total_steps % opt.display_freq == display_delta

    ############## Forward Pass ######################
    # print(pair)
    generated = model(paddle.to_tensor(data['label'], stop_gradient=False),
                      paddle.to_tensor(data['inst'], stop_gradient=False))

    # sum per device losses
    performance.update(generated, data['image'])
    PSNR, SSIM, FID, LPIPS = performance.accumulate()
    performance.reset()
    visualizer.print_current_performance(0, PSNR, SSIM, FID, LPIPS)

    performance_all.update(generated[:5], data['image'][:5])
    ############## Display results and errors ##########

    ### display output images
    if save_fake:
        if not os.path.exists(opt.outputs_dir + opt.name):
            os.makedirs(opt.outputs_dir + opt.name)
        imgs_num = 5
        if opt.NL_use_mask:
            mask = data['inst'][:imgs_num]
            mask = mask.repeat(1, 3, 1, 1)
            imgs = paddle.concat(
                (data['label'][:imgs_num], mask, generated[:imgs_num], data['image'][:imgs_num]), 0)
        else:
            imgs = paddle.concat(
                (data['label'][:imgs_num], generated[:imgs_num], data['image'][:imgs_num]), 0)
        imgs = (imgs + 1.) / 2.0  ## de-normalize

        try:
            vutils.save_image(imgs, opt.outputs_dir + opt.name + '_' + str(
                total_steps) + '.png', nrow=imgs_num, padding=0, normalize=True)
        except OSError as err:
            print(err)

    if epoch_iter >= dataset_size:
        break

PSNR, SSIM, FID, LPIPS = performance_all.accumulate()
performance_all.reset()
visualizer.print_current_performance(0, PSNR, SSIM, FID, LPIPS)
# end of epoch
epoch_e_t = datetime.datetime.now()
iter_end_time = time.time()
visualizer.print_log('End ====== :\t Time Taken: %s' % (str(epoch_e_t - epoch_s_t)))
