from collections import OrderedDict
from datetime import datetime
import logging
from math import ceil
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import SubsetRandomSampler
from logger import Logger
from constants import TrainingConstants
from losses import compute_diou
import wandb

class Trainer(object):
    def __init__(self,
                 feature_net,
                 bounding_box_net,
                 dataset,
                 num_epochs=TrainingConstants.NUM_EPOCHS,
                 total_size=TrainingConstants.TOTAL_SIZE,
                 val_size=TrainingConstants.VAL_SIZE,
                 bsz=TrainingConstants.BSZ,
                 feature_base_lr=TrainingConstants.FEATURE_BASE_LR,
                 feature_lr_step_size=TrainingConstants.FEATURE_LR_STEP_SIZE,
                 feature_lr_decay_rate=TrainingConstants.FEATURE_LR_DECAY_RATE,
                 bb_base_lr=TrainingConstants.BB_BASE_LR,
                 bb_lr_step_size=TrainingConstants.BB_LR_STEP_SIZE,
                 bb_lr_decay_rate=TrainingConstants.BB_LR_DECAY_RATE,
                 log_interval=TrainingConstants.LOG_INTERVAL,
                 device=TrainingConstants.DEVICE,
                 output_dir=TrainingConstants.OUTPUT_DIR):
        self._feature_net = feature_net
        self._bb_net = bounding_box_net
        self._dataset = dataset
        self._num_epochs = num_epochs
        self._total_size = total_size
        self._val_size = val_size
        self._bsz = bsz
        self._feature_base_lr = feature_base_lr
        self._feature_lr_step_size = feature_lr_step_size
        self._feature_lr_decay_rate = feature_lr_decay_rate
        self._bb_base_lr = bb_base_lr
        self._bb_lr_step_size = bb_lr_step_size
        self._bb_lr_decay_rate = bb_lr_decay_rate
        self._device = device
        self._log_interval = log_interval
        self._output_dir = output_dir

        self._native_logger = logging.getLogger(self.__class__.__name__)

    def _setup(self):
        date_time = datetime.now().strftime('%m-%d-%Y_%H:%M:%S')
        self._output_dir = os.path.join(
            self._output_dir,
            '{}_{}_{}'.format(self._feature_net.name, self._bb_net.name, str(date_time))
        )
        os.makedirs(self._output_dir)

        self._logger = Logger(
            os.path.join(self._output_dir, TrainingConstants.LOG_DIR)
        )

        ind = np.arange(len(self._dataset))
        # np.random.shuffle(ind)
        ind = ind[:ceil(self._total_size*len(ind))]
        train_ind = ind[:ceil((1-self._val_size)*len(ind))]
        val_ind = ind[ceil((1-self._val_size)*len(ind)):]

        train_sampler = SubsetRandomSampler(train_ind)
        val_sampler = SubsetRandomSampler(val_ind)
        self._train_data_loader = torch.utils.data.DataLoader(
                                    self._dataset,
                                    batch_size=self._bsz,
                                    num_workers=3,
                                    pin_memory=True,
                                    sampler=train_sampler
                                 )
        self._val_data_loader = torch.utils.data.DataLoader(
                                    self._dataset,
                                    batch_size=self._bsz,
                                    num_workers=3,
                                    pin_memory=True,
                                    sampler=val_sampler
                               )

        self._device = torch.device(self._device)
        # self._feature_net = torch.nn.DataParallel(self._feature_net, device_ids=[0,1])
        # self._bb_net = torch.nn.DataParallel(self._bb_net, device_ids=[0, 1])
        self._feature_net.to(self._device)
        self._bb_net.to(self._device)

        #self._feature_optimizer = torch.optim.Adam(self._feature_net.parameters(), lr=self._feature_base_lr, betas=(TrainingConstants.ADAM_BETA1, TrainingConstants.ADAM_BETA2))
        self._feature_optimizer = torch.optim.Adadelta(self._feature_net.parameters(), lr=self._feature_base_lr)
        self._feature_scheduler = StepLR(self._feature_optimizer, step_size=self._feature_lr_step_size, gamma=self._feature_lr_decay_rate)
        self._bb_optimizer = torch.optim.Adam(self._bb_net.parameters(), lr=self._bb_base_lr, betas=(TrainingConstants.ADAM_BETA1, TrainingConstants.ADAM_BETA2))
        # self._bb_optimizer = torch.optim.Adadelta(self._bb_net.parameters(), lr=self._bb_base_lr)
        self._bb_scheduler = StepLR(self._bb_optimizer, step_size=self._bb_lr_step_size, gamma=self._bb_lr_decay_rate)
        
        self._feature_criterion = nn.L1Loss(reduction="none")
        self._bb_criterion = nn.SmoothL1Loss(reduction="mean")

        # Use wandb to visualize bounding boxes
        wandb.init(project="vlr-project", reinit=True)


    def _log_metric(self, epoch, metric_name, data):
        self._native_logger.info('Logging {} ...'.format(metric_name))

        if not isinstance(data, (list, np.ndarray)):
            data = [data]
        data = np.asarray(data)

        logs = OrderedDict()
        logs['{}_average'.format(metric_name)] = np.mean(data)
        logs['{}_stddev'.format(metric_name)] = np.std(data)
        logs['{}_max'.format(metric_name)] = np.max(data)
        logs['{}_min'.format(metric_name)] = np.min(data)

        # Write Tensorboard summaries
        for key, value in logs.items():
            self._native_logger.info('\t{} : {}'.format(key, value))
            self._logger.log_scalar(value, key, epoch)
        self._logger.flush()


    def _train(self, epoch):
        self._feature_net.train()

        num_batches = len(self._train_data_loader)
        feature_train_losses = []
        bb_train_losses = []

        for batch_idx, (scene_img, target_img, bb, bb_original) in enumerate(self._train_data_loader):
            scene_img = scene_img.to(self._device)
            target_img = target_img.to(self._device)
            bb = bb.to(self._device)
            bb_original = bb_original.to(self._device)
            
            self._feature_optimizer.zero_grad()
            
            bb_output = self._feature_net((scene_img, target_img, bb_original))
            print('pred: {}, gt: {}'.format(bb_output[-1], bb[-1]))
            bb_loss = self._bb_criterion(bb_output, bb)
            # iou, diou = compute_diou(bb_output, bb)
            # bb_loss += 10 * iou
            bb_loss.backward()
            self._feature_optimizer.step()

            if batch_idx % self._log_interval == 0:
                self._native_logger.info(
                    'Train Epoch: {} [Batch {}/{} ({:.0f}%)]\tBB Loss: {:.6f}\t'
                    'Feature LR: {:.6f}'.format(
                        epoch,
                        batch_idx+1,
                        num_batches,
                        100 * (batch_idx+1) / num_batches,
                        bb_loss.item(),
                        self._feature_optimizer.param_groups[0]['lr'],
                    )
                )
                bb_train_losses.append(bb_loss.item())

            # for i in range(scene_img.shape[0]):
            for i in range(5):
                pred_bb = bb_output[i]
                x_min = max(0, pred_bb[0].item())
                x_max = min(1.0, pred_bb[0].item()+pred_bb[2].item())
                y_min = max(0, pred_bb[1].item())
                y_max = min(1.0, pred_bb[1].item()+pred_bb[3].item())

                # Plot image to wandb
                scene_wandb = scene_img[i].cpu().detach().numpy()
                scene_wandb = scene_wandb.transpose(1, 2, 0)
                wandb.log({"image_proposals": wandb.Image(scene_wandb, boxes={
                    "proposals": {
                        "box_data": [{"position": {"minX": x_min, "minY": y_min, "maxX": x_max, "maxY": y_max}, "class_id": 0}],
                        "class_labels": {0: "prediction"}
                    }
                })})
                gt_x_min, gt_y_min, gt_x_max, gt_y_max = bb[i].cpu().detach().numpy()
                gt_x_max += gt_x_min
                gt_y_max += gt_y_min 
                wandb.log({"image_gt": wandb.Image(scene_wandb, boxes={
                    "ground_truth": {
                        "box_data": [{"position": {"minX": gt_x_min, "minY": gt_y_min, "maxX": gt_x_max, "maxY": gt_y_max}, "class_id": 0}],
                        "class_labels": {0: "ground_truth"}
                    }
                })})

        self._log_metric(epoch, 'train/epoch_bb_loss', bb_train_losses)

    def _eval(self, epoch):
        self._feature_net.eval()

        feature_eval_losses = []
        bb_eval_losses = []
        with torch.no_grad():
            for batch_idx, (scene_img, target_img, bb, bb_original) in enumerate(self._val_data_loader):
                scene_img = scene_img.to(self._device)
                target_img = target_img.to(self._device)
                bb = bb.to(self._device)
                bb_original = bb_original.to(self._device)
                
                bb_output = self._feature_net((scene_img, target_img, bb_original))
                bb_loss = self._bb_criterion(bb_output, bb)
                bb_eval_losses.append(bb_loss.item())

        self._log_metric(epoch, 'eval/epoch_bb_loss', bb_eval_losses)

    def train(self):
        self._setup()
        for epoch in range(1, self._num_epochs+1):
            self._train(epoch)
            self._eval(epoch)
            self._feature_scheduler.step()

            self._native_logger.info('')
            if epoch % TrainingConstants.NET_SAVE_FREQUENCY == 0:
                self._feature_net.save(self._output_dir, TrainingConstants.FEATURE_NET_SAVE_FNAME, str(epoch) + '_')
        
        self._feature_net.save(self._output_dir, TrainingConstants.FEATURE_NET_SAVE_FNAME, 'final_')
