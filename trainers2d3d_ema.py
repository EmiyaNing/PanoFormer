from __future__ import absolute_import, division, print_function
import os

import numpy as np
import time
import json
import tqdm

import cv2

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
torch.manual_seed(100)
torch.cuda.manual_seed(100)

from metrics import compute_depth_metrics, Evaluator
from losses import BerhuLoss
import loss_gradient as loss_g
# from network.Decoder import FCRNDecoder as PanoBiT
from network.model import Panoformer as PanoBiT
from stanford2d3d import Stanford2D3D

from ema_pytorch import EMA


def gradient(x):
    gradient_model = loss_g.Gradient_Net()
    g_x, g_y = gradient_model(x)
    return g_x, g_y


class EMA_Trainer:
    def __init__(self, settings):
        self.settings = settings

        self.device = torch.device("cuda" if len(self.settings.gpu_devices) else "cpu")
        self.gpu_devices = ','.join([str(id) for id in settings.gpu_devices])
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_devices

        self.log_path = os.path.join(self.settings.log_dir, self.settings.model_name)
        self.ema_val  = settings.ema_val

        # data
        train_dataset = Stanford2D3D('../data/train/','./splits2d3d/stanford2d3d_train.txt', 
                                     disable_color_augmentation=self.settings.disable_color_augmentation, 
                                     disable_LR_filp_augmentation=self.settings.disable_LR_filp_augmentation,
                                     disable_yaw_rotation_augmentation=self.settings.disable_yaw_rotation_augmentation, 
                                     is_training=True)
                                     

        self.train_loader = DataLoader(train_dataset, self.settings.batch_size, True,
                                       num_workers=self.settings.num_workers, pin_memory=True, drop_last=True)
        num_train_samples = len(train_dataset)
        self.num_total_steps = num_train_samples // self.settings.batch_size * self.settings.num_epochs

        val_dataset = Stanford2D3D('../data/val/','./splits2d3d/stanford2d3d_test.txt', 
                                    disable_color_augmentation=self.settings.disable_color_augmentation,
                                    disable_LR_filp_augmentation=self.settings.disable_LR_filp_augmentation,
                                    disable_yaw_rotation_augmentation=self.settings.disable_yaw_rotation_augmentation, 
                                    is_training=False)#self.dataset(self.settings.data_path, val_file_list, self.settings.height, self.settings.width,
                                   # self.settings.disable_color_augmentation, self.settings.disable_LR_filp_augmentation,
                                   # self.settings.disable_yaw_rotation_augmentation, is_training=False)
        self.val_loader = DataLoader(val_dataset, self.settings.batch_size, False,
                                     num_workers=self.settings.num_workers, pin_memory=True, drop_last=True)

        self.model = PanoBiT()
        self.model.to(self.device)
        self.ema   = EMA(self.model, beta=0.9999, update_after_step = 100, update_every = 5)
        import pdb
        pdb.set_trace()


        self.parameters_to_train = list(self.model.parameters())

        self.optimizer = optim.Adam(self.parameters_to_train, self.settings.learning_rate)

        if self.settings.load_weights_dir is not None:
            self.load_model()

        print("Training model named:\n ", self.settings.model_name)
        print("Models and tensorboard events files are saved to:\n", self.settings.log_dir)
        print("Training is using:\n ", self.device)

        self.compute_loss = BerhuLoss()
        self.evaluator = Evaluator()

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.save_settings()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        self.validate()
        for self.epoch in range(self.settings.num_epochs):
            self.train_one_epoch()
            if (self.epoch + 1) % self.settings.save_frequency == 0:
                self.save_model()
            self.validate()
            

    def train_one_epoch(self):
        """Run a single epoch of training
        """
        self.model.train()

        pbar = tqdm.tqdm(self.train_loader)
        pbar.set_description("Training Epoch_{}".format(self.epoch))

        for batch_idx, inputs in enumerate(pbar):

            outputs, losses = self.process_batch(inputs)

            self.optimizer.zero_grad()
            losses["loss"].backward()
            self.optimizer.step()
            self.ema.update()

            # log less frequently after the first 1000 steps to save time & disk space
            early_phase = batch_idx % self.settings.log_frequency == 0 and self.step < 1000
            late_phase = self.step % 1000 == 0

            if early_phase or late_phase:

                pred_depth = outputs["pred_depth"].detach()
                gt_depth = inputs["gt_depth"]
                mask = inputs["val_mask"]
                
                pred_depth = outputs["pred_depth"].detach() * mask
                gt_depth = inputs["gt_depth"] * mask
                

                depth_errors = compute_depth_metrics(gt_depth, pred_depth, mask)
                for i, key in enumerate(self.evaluator.metrics.keys()):
                    losses[key] = np.array(depth_errors[i].cpu())

                self.log("train", inputs, outputs, losses)

            self.step += 1

    def process_batch(self, inputs):
        for key, ipt in inputs.items():
            if key not in ["rgb"]:
                inputs[key] = ipt.to(self.device)

        losses = {}

        equi_inputs = inputs["normalized_rgb"] * inputs["val_mask"]


        outputs = self.model(equi_inputs)

        gt = inputs["gt_depth"] * inputs["val_mask"]
        pred = outputs["pred_depth"] * inputs["val_mask"]
        outputs["pred_depth"] = outputs["pred_depth"] * inputs["val_mask"]

        loss_weight_mask = torch.ones([512, 1024],device=gt.device, dtype=pred.dtype)
        # the black region's loss will be seted to zeros
        loss_weight_mask[0:int(512 * 0.15), :] = 0
        loss_weight_mask[512 - int(512 * 0.15):512, :] = 0
        # as the floor and ceil is easy to learn we reduce it's importance
        # during the training process
        loss_weight_mask[int(512 * 0.15):int(512 * 0.25), :] = 0.5
        loss_weight_mask[512 - int(512*0.25): 512 - int(512 * 0.15), :] = 0.5


        G_x, G_y = gradient(gt.float())
        p_x, p_y = gradient(pred)

        loss_gt_depth   = self.compute_loss(inputs["gt_depth"].float() * inputs["val_mask"] * loss_weight_mask, outputs["pred_depth"] * loss_weight_mask)
        loss_x_gradient = self.compute_loss(G_x * loss_weight_mask, p_x * loss_weight_mask)
        loss_y_gradient = self.compute_loss(G_y * loss_weight_mask, p_y * loss_weight_mask)
        losses["loss"]  = loss_gt_depth + loss_x_gradient + loss_y_gradient

        return outputs, losses

    def process_batch_ema(self, inputs):
        '''
            This function only used into the validate process.
            In this function we only use the ema model to get the outputs.
        '''
        for key, ipt in inputs.items():
            if key not in ["rgb"]:
                inputs[key] = ipt.to(self.device)
        
        losses = {}


        equi_inputs           = inputs["normalized_rgb"] * inputs["val_mask"]
        outputs               = self.ema(equi_inputs)
        gt                    = inputs["gt_depth"] * inputs["val_mask"]
        pred                  = outputs["pred_depth"] * inputs["val_mask"]
        outputs["pred_depth"] = outputs["pred_depth"] * inputs["val_mask"]
        loss_weight_mask = torch.ones([512, 1024],device=gt.device, dtype=pred.dtype)
        # the black region's loss will be seted to zeros
        loss_weight_mask[0:int(512 * 0.15), :] = 0
        loss_weight_mask[512 - int(512 * 0.15):512, :] = 0
        # as the floor and ceil is easy to learn we reduce it's importance
        # during the training process
        loss_weight_mask[int(512 * 0.15):int(512 * 0.25), :] = 0.5
        loss_weight_mask[512 - int(512*0.25): 512 - int(512 * 0.15), :] = 0.5


        G_x, G_y = gradient(gt.float())
        p_x, p_y = gradient(pred)

        loss_gt_depth   = self.compute_loss(inputs["gt_depth"].float() * inputs["val_mask"] * loss_weight_mask, outputs["pred_depth"] * loss_weight_mask)
        loss_x_gradient = self.compute_loss(G_x * loss_weight_mask, p_x * loss_weight_mask)
        loss_y_gradient = self.compute_loss(G_y * loss_weight_mask, p_y * loss_weight_mask)
        losses["loss"]  = loss_gt_depth + loss_x_gradient + loss_y_gradient

        return outputs, losses



    def validate(self):
        """Validate the model on the validation set
        """
        self.model.eval()
        self.ema.eval()

        self.evaluator.reset_eval_metrics()

        pbar = tqdm.tqdm(self.val_loader)
        pbar.set_description("Validating Epoch_{}".format(self.epoch))

        with torch.no_grad():
            for batch_idx, inputs in enumerate(pbar):
                if not self.ema_val:
                    outputs, losses = self.process_batch(inputs)
                else:
                    outputs, losses = self.process_batch_ema(inputs)
                pred_depth = outputs["pred_depth"].detach() * inputs["val_mask"]
                gt_depth = inputs["gt_depth"] * inputs["val_mask"]

                # follows code will be used to store the visualize the model's predictions.
                pred_depth1 = (pred_depth[0] * 512).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint16)
                pred_depth2 = (pred_depth[1] * 512).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint16)


                gt_depth1 = (gt_depth[0] * 512).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint16)
                gt_depth2 = (gt_depth[1] * 512).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint16)


                rgb1 = (inputs["rgb"][0].permute(1, 2, 0) * 255).detach().cpu().numpy().astype(np.uint8)
                rgb2 = (inputs['rgb'][1].permute(1, 2, 0) * 255).detach().cpu().numpy().astype(np.uint8)

                cv2.imwrite('./val_rgb/' + str(batch_idx* 2 + 0) + ".png", rgb1)
                cv2.imwrite('./val_rgb/' + str(batch_idx* 2 + 1) + ".png", rgb2)


                cv2.imwrite('./pred_depth/' + str(batch_idx*2 + 0) + ".png", pred_depth1)
                cv2.imwrite('./pred_depth/' + str(batch_idx*2 + 1) + ".png", pred_depth2)

                cv2.imwrite('./gt_depth/' + str(batch_idx*2 + 0) + ".png", gt_depth1)
                cv2.imwrite('./gt_depth/' + str(batch_idx*2 + 1) + ".png", gt_depth2)

                self.evaluator.compute_eval_metrics(gt_depth, pred_depth)

        self.evaluator.print()

        for i, key in enumerate(self.evaluator.metrics.keys()):
            losses[key] = np.array(self.evaluator.metrics[key].avg.cpu())
        self.log("val", inputs, outputs, losses)
        del inputs, outputs, losses

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        outputs["pred_depth"] = outputs["pred_depth"] * inputs["val_mask"]
        inputs["gt_depth"] = inputs["gt_depth"] * inputs["val_mask"]
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.settings.batch_size)):  # write a maxmimum of four images
            writer.add_image("rgb/{}".format(j), inputs["rgb"][j].data, self.step)
            # writer.add_image("cube_rgb/{}".format(j), inputs["cube_rgb"][j].data, self.step)
            writer.add_image("gt_depth/{}".format(j),
                             inputs["gt_depth"][j].data/inputs["gt_depth"][j].data.max(), self.step)
            writer.add_image("pred_depth/{}".format(j),
                             outputs["pred_depth"][j].data/outputs["pred_depth"][j].data.max(), self.step)

    def save_settings(self):
        """Save settings to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.settings.__dict__.copy()

        with open(os.path.join(models_dir, 'settings.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_path = os.path.join(save_folder, "{}.pth".format("model"))
        to_save = self.model.state_dict()
        torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("ema"))
        to_save   = self.ema.ema_model.state_dict()
        torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model from disk
        """
        self.settings.load_weights_dir = os.path.expanduser(self.settings.load_weights_dir)

        assert os.path.isdir(self.settings.load_weights_dir), \
            "Cannot find folder {}".format(self.settings.load_weights_dir)
        print("loading model from folder {}".format(self.settings.load_weights_dir))

        path = os.path.join(self.settings.load_weights_dir, "{}.pth".format("model"))
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.settings.load_weights_dir, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")


