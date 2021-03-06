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


np.random.seed(1337)

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
        np.random.shuffle(ind)
        ind = ind[:ceil(self._total_size*len(ind))]
        train_ind = ind[:ceil((1-self._val_size)*len(ind))]
        val_ind = ind[ceil((1-self._val_size)*len(ind)):]

        #train_ind = ind[:1]
        #val_ind = ind[:1]

        train_sampler = SubsetRandomSampler(train_ind)
        val_sampler = SubsetRandomSampler(val_ind)
        self._train_data_loader = torch.utils.data.DataLoader(
                                    self._dataset,
                                    batch_size=self._bsz,
                                    num_workers=8,
                                    pin_memory=True,
                                    sampler=train_sampler,
                                    drop_last=True
                                 )
        self._val_data_loader = torch.utils.data.DataLoader(
                                    self._dataset,
                                    batch_size=self._bsz,
                                    num_workers=1,
                                    pin_memory=True,
                                    sampler=val_sampler,
                                    drop_last=True
                               )

        self._device = torch.device(self._device)
        self._feature_net = torch.nn.DataParallel(self._feature_net, device_ids=[0,1,2,3])
        self._bb_net = torch.nn.DataParallel(self._bb_net, device_ids=[0,1,2,3])
        self._feature_net.to(self._device)
        self._bb_net.to(self._device)

        self._feature_optimizer = torch.optim.Adam(self._feature_net.parameters(), lr=self._feature_base_lr) #, betas=(TrainingConstants.ADAM_BETA1, TrainingConstants.ADAM_BETA2))
        self._feature_scheduler = StepLR(self._feature_optimizer, step_size=self._feature_lr_step_size, gamma=self._feature_lr_decay_rate)
        self._bb_optimizer = torch.optim.Adam(self._bb_net.parameters(), lr=self._bb_base_lr) #, betas=(TrainingConstants.ADAM_BETA1, TrainingConstants.ADAM_BETA2))
        self._bb_scheduler = StepLR(self._bb_optimizer, step_size=self._bb_lr_step_size, gamma=self._bb_lr_decay_rate)
        
        self._feature_criterion = nn.MSELoss()
        self._bb_criterion = nn.SmoothL1Loss(reduction='mean')

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

    def crop(self, image, template):
        y_nonzero, x_nonzero, _ = np.nonzero(template.cpu().detach().numpy())
        return image[:,np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]


    def _train(self, epoch):
        self._feature_net.train()
        self._bb_net.train()

        num_batches = len(self._train_data_loader)
        feature_train_losses = []
        bb_train_losses = []

<<<<<<< HEAD
        for batch_idx, (scene_img, target_img, bb,_) in enumerate(self._train_data_loader):
=======
        for batch_idx, (scene_img, target_img, bb, crop_coords) in enumerate(self._train_data_loader):
>>>>>>> ac6cd9ffe60ebaade69544ffecb4ef9e2eb9215b
            scene_img = scene_img.to(self._device)
            target_img = target_img.to(self._device)

            #target_template = target_template.to(self._device)
            bb = bb.to(self._device)

            self._feature_optimizer.zero_grad()
            self._bb_optimizer.zero_grad()
            
<<<<<<< HEAD

            feature_output = self._feature_net((scene_img, target_img))
=======
            ''' DOT PRODUCT FOR FEATURE NET'''
            # feature_output = self._feature_net((scene_img, target_img))
            ''' CONVOLUTION FOR FEATURE NET '''
            scene_features = self._feature_net.forward_scene(scene_img)
            target_features = self._feautre_net.forward_target(target_img)
            # Crop target features using crop coordinates
            target_features = target_features[:, :, crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3]]
            feature_output = F.conv2d(scene_features, target_features)

>>>>>>> ac6cd9ffe60ebaade69544ffecb4ef9e2eb9215b
            bb_output = self._bb_net(feature_output)
            print(bb_output, bb)
            #bb_output[:,2] += bb_output[:,0]
            #bb_output[:,3] += bb_output[:,1]

            ''' Compute bounding box loss using bounding box regression loss. '''
            # bb loss: https://towardsdatascience.com/bounding-box-prediction-from-scratch-using-pytorch-a8525da51ddc
<<<<<<< HEAD
            bb_loss = 100*self._bb_criterion(bb_output, bb.float())
            #iou, diou = compute_diou(bb_output.float(), bb.float())
            #bb_loss += iou
=======
            bb_output_ = bb_output.clone()
            bb_ = bb.clone()
            bb_output_[:,2] += bb_output_[:,0]
            bb_output_[:,3] += bb_output_[:,1]
            bb_[:,2] += bb_[:,0]
            bb_[:,3] += bb_[:,1]

            bb_loss = self._bb_criterion(bb_output, bb).sum(1).mean()
            #_, diou = compute_diou(bb_output, bb)
            #bb_loss += 10*diou
>>>>>>> ac6cd9ffe60ebaade69544ffecb4ef9e2eb9215b
            bb_loss.backward()
            self._bb_optimizer.step()
            self._feature_optimizer.step()

            ''' Compute feature net loss '''
            # # Get target feature map from target feature extractor head
            # # Get predicted bounding box feature map from scene feature extractor head. Modify scene_img inplace to save memory?
            # # TODO: can we vectorize this?
            feature_loss = torch.tensor([-1])

<<<<<<< HEAD
            if epoch > 100:
                #self._feature_optimizer.zero_grad() # Zero out feature output gradients again just in case?
                #self._bb_optimizer.zero_grad()

                bb_features = []
                target_features = []
                feature_loss = 0
                target_feat = self._feature_net.module.forward_scene(target_img)
                for j in range(bb_output.shape[0]):
                    #pred_bb = bb_output[j]
                    pred_bb = bb[j]
                    x_min = int(np.floor(max(0, pred_bb[0].item()*scene_img.shape[3])))  # scene_img : [B, C, H, W]
                    width = int(np.ceil(min(scene_img.shape[3], pred_bb[2].item()*scene_img.shape[3])))
                    x_max = int(np.ceil(min(scene_img.shape[3], x_min + width)))
                    # x_max = int(np.ceil(min(scene_img.shape[2], pred_bb[1].item() + 1)))
                    y_min = int(np.floor(max(0, pred_bb[1].item()*scene_img.shape[2])))
                    height = int(np.ceil(min(scene_img.shape[2], pred_bb[3].item()*scene_img.shape[2])))
                    y_max = int(np.ceil(min(scene_img.shape[2], y_min + height)))
                    # y_max = int(np.ceil(min(scene_img.shape[3], pred_bb[3].item() + 1)))
                    #print(x_min, x_max, y_min, y_max, pred_bb, bb[j])
                    
                    bb_feat = scene_feat[j, :, y_min:y_max, x_min:x_max].unsqueeze(0) #, size=(scene_img.shape[2], scene_img.shape[3]), mode='bilinear')
                  
                    with torch.no_grad():
                        target_crop_feat = self.crop(target_feat[j], target_template[j]).unsqueeze(0)
                        target_crop_feat = F.interpolate(target_crop_feat, size=(bb_feat.shape[2], bb_feat.shape[3]))

                    feature_loss += torch.mean((bb_feat - target_crop_feat)**2)
              
                #target_features = torch.stack(target_features)
                #bb_features = torch.stack(bb_features)

                #feature_loss = self._feature_criterion(bb_features, target_features)
                feature_loss.backward()
                self._feature_optimizer.step()
                #self._bb_optimizer.zero_grad()
=======
            if epoch > 1:
              for i in range(len(bb_output)):
                  pred_bb = bb_output[i]
                  x_min = int(np.floor(max(0, pred_bb[0].item()*scene_img.shape[3])))  # scene_img : [B, C, H, W]
                  width = int(np.ceil(max(0, pred_bb[2].item()*scene_img.shape[3])))
                  x_max = int(np.ceil(min(scene_img.shape[3], x_min + width + 1)))
                  # x_max = int(np.ceil(min(scene_img.shape[2], pred_bb[1].item() + 1)))
                  y_min = int(np.floor(max(0, pred_bb[1].item()*scene_img.shape[2])))
                  height = int(np.ceil(max(0, pred_bb[3].item()*scene_img.shape[2])))
                  y_max = int(np.ceil(min(scene_img.shape[2], y_min + height + 1)))
                  # y_max = int(np.ceil(min(scene_img.shape[3], pred_bb[3].item() + 1)))
                  try:
                      scene_img[i] = F.interpolate(scene_img[i, :, y_min:y_max, x_min:x_max].unsqueeze(0), size=(scene_img.shape[2], scene_img.shape[3]), mode='bilinear')
                  except:
                      print(x_min, x_max, y_min, y_max, pred_bb, scene_img.shape)

              bb_feature_map = self._feature_net.module.forward_scene(scene_img)
              feature_loss = self._feature_criterion(target_feature_map, bb_feature_map).sum(1).mean()
              feature_loss.backward()
              self._feature_optimizer.step()
>>>>>>> ac6cd9ffe60ebaade69544ffecb4ef9e2eb9215b


            if batch_idx % self._log_interval == 0:
                self._native_logger.info(
                    'Train Epoch: {} [Batch {}/{} ({:.0f}%)]\tFeature Loss: {:.6f}\tBB Loss: {:.6f}\t'
                    'Feature LR: {:.6f}\t BB LR: {:.6f}'.format(
                        epoch,
                        batch_idx+1,
                        num_batches,
                        100 * (batch_idx+1) / num_batches,
                        feature_loss.item(),
                        bb_loss.item(),
                        self._feature_optimizer.param_groups[0]['lr'],
                        self._bb_optimizer.param_groups[0]['lr']
                    )
                )
                feature_train_losses.append(feature_loss.item())
                bb_train_losses.append(bb_loss.item())

        self._log_metric(epoch, 'train/epoch_feature_loss', feature_train_losses)
        self._log_metric(epoch, 'train/epoch_bb_loss', bb_train_losses)

    def _eval(self, epoch):
        self._feature_net.eval()
        self._bb_net.eval()

        feature_eval_losses = []
        bb_eval_losses = []
        with torch.no_grad():
            i = 0
            for batch_idx, (scene_img, target_img, bb, crop_coords) in enumerate(self._val_data_loader):
                i += 1
                scene_img = scene_img.to(self._device)
                target_img = target_img.to(self._device)
                bb = bb.to(self._device)
                
                ''' DOT PRODUCT FOR FEATURE NET'''
                # feature_output = self._feature_net((scene_img, target_img))
                ''' CONVOLUTION FOR FEATURE NET '''
                scene_features = self._feature_net.forward_scene(scene_img)
                target_features = self._feautre_net.forward_target(target_img)
                # Crop target features using crop coordinates
                target_features = target_features[:, :, crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3]]
                feature_output = F.conv2d(scene_features, target_features)
                
                bb_output = self._bb_net(feature_output)
                #bb_output[:,2] += bb_output[:,0]
                #bb_output[:,3] += bb_output[:,1]

                bb_loss = self._bb_criterion(bb_output, bb)

                target_feature_map = self._feature_net.module.forward_target(target_img)

                feature_loss = torch.tensor([-1])

                if epoch > 10:

<<<<<<< HEAD
                  for j in range(len(bb_output)):
                      pred_bb = bb_output[j]
                      x_min = int(np.floor(max(0, pred_bb[0].item()*scene_img.shape[3])))
                      x_max = int(np.ceil(min(scene_img.shape[3], pred_bb[2].item()*scene_img.shape[3])))
                      #x_max = int(np.ceil(min(scene_img.shape[2], x_min + width + 1)))
                      # x_max = int(np.ceil(min(scene_img.shape[2], pred_bb[1].item() + 1)))
                      y_min = int(np.floor(max(0, pred_bb[1].item()*scene_img.shape[2])))
                      y_max = int(np.ceil(min(scene_img.shape[2], pred_bb[3].item()*scene_img.shape[2])))
                      #y_max = int(np.ceil(min(scene_img.shape[3], y_min + height + 1)))
                      # y_max = int(np.ceil(min(scene_img.shape[3], pred_bb[3].item() + 1)))
                      scene_img[j] = F.interpolate(scene_img[j, :, y_min:y_max, x_min:x_max].unsqueeze(0), size=(scene_img.shape[2], scene_img.shape[3]), mode='bilinear')
=======
                  for i in range(len(bb_output)):
                      pred_bb = bb_output[i]
                      x_min = int(np.floor(max(0, pred_bb[0].item())))
                      width = int(np.ceil(max(0, pred_bb[2].item())))
                      x_max = int(np.ceil(min(scene_img.shape[3], x_min + width + 1)))
                      # x_max = int(np.ceil(min(scene_img.shape[2], pred_bb[1].item() + 1)))
                      y_min = int(np.floor(max(0, pred_bb[1].item())))
                      height = int(np.ceil(max(0, pred_bb[3].item())))
                      y_max = int(np.ceil(min(scene_img.shape[2], y_min + height + 1)))
                      # y_max = int(np.ceil(min(scene_img.shape[3], pred_bb[3].item() + 1)))
                      scene_img[i] = F.interpolate(scene_img[i, :, y_min:y_max, x_min:x_max].unsqueeze(0), size=(scene_img.shape[2], scene_img.shape[3]), mode='bilinear')
>>>>>>> ac6cd9ffe60ebaade69544ffecb4ef9e2eb9215b
                  bb_feature_map = self._feature_net.module.forward_scene(scene_img)
                  feature_loss = self._feature_criterion(target_feature_map, bb_feature_map)

                feature_eval_losses.append(feature_loss.item())
                bb_eval_losses.append(bb_loss.item())

        # self._log_metric(epoch, 'eval/epoch_feature_loss', feature_eval_losses)
        self._log_metric(epoch, 'eval/epoch_bb_loss', bb_eval_losses)

    def train(self):
        self._setup()
        for epoch in range(1, self._num_epochs+1):
            self._train(epoch)
            #self._eval(epoch)
            self._feature_scheduler.step()
            self._bb_scheduler.step()

            self._native_logger.info('')
            if epoch % TrainingConstants.NET_SAVE_FREQUENCY == 0:
                self._feature_net.module.save(self._output_dir, TrainingConstants.FEATURE_NET_SAVE_FNAME, str(epoch) + '_')
                self._bb_net.module.save(self._output_dir, TrainingConstants.BOUNDING_BOX_NET_SAVE_FNAME, str(epoch) + '_')
        
        self._feature_net.module.save(self._output_dir, TrainingConstants.FEATURE_NET_SAVE_FNAME, 'final_')
        self._bb_net.module.save(self._output_dir, TrainingConstants.BOUNDING_BOX_NET_SAVE_FNAME, 'final_')
