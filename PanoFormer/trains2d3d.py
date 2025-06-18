from __future__ import absolute_import, division, print_function
import os
import argparse

from trainers2d3d import Trainer
from trainers2d3d_ema import EMA_Trainer
from trainers2d3d_ema_mult import EMA_Trainer_Mult
from trainers2d3d_onecycle import EMA_Trainer_Onecycle

parser = argparse.ArgumentParser(description="360 Degree Panorama Depth Estimation Training")

# system settings
parser.add_argument("--num_workers", type=int, default=8, help="number of dataloader workers")
parser.add_argument("--gpu_devices", type=int, nargs="+", default=[0], help="available gpus")

# model settings
parser.add_argument("--model_name", type=str, default="panodepth", help="folder to save the model in")

# optimization settings
parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
parser.add_argument("--batch_size", type=int, default=2, help="batch size")
parser.add_argument("--num_epochs", type=int, default=20, help="number of epochs")

# loading and logging settings
parser.add_argument("--load_weights_dir", default='./tmp_s2d3d/panodepth/models/weights', type=str, help="folder of model to load")#, default='./tmp_abl_offset/panodepth/models/weights_49'
#parser.add_argument("--load_weights_dir", default='./tmp_s2d3d/panodepth/best_01_2465/weights_1', type=str)
#parser.add_argument("--load_weights_dir", default=None, type=str)
parser.add_argument("--log_dir", type=str, default=os.path.join(os.path.dirname(__file__), "ema_panoformer"), help="log directory")
parser.add_argument("--log_frequency", type=int, default=1, help="number of batches between each tensorboard log")
parser.add_argument("--save_frequency", type=int, default=1, help="number of epochs between each save")
parser.add_argument("--ema", action="store_true", help="This parameter used to judge whether use ema training")
parser.add_argument("--ema_val", action="store_true", help="This parameter is used to judge whether use ema to validate")
parser.add_argument("--one_cycle", action="store_true", help="Use this to judge whether use one_cycle to determine the learning rate.")
parser.add_argument("--mult", action="store_true", help="This parameter is used to configure mult-stage training")

# data augmentation settings
parser.add_argument("--disable_color_augmentation", action="store_true", help="if set, do not use color augmentation")
parser.add_argument("--disable_LR_filp_augmentation", action="store_true",
                    help="if set, do not use left-right flipping augmentation")
parser.add_argument("--disable_yaw_rotation_augmentation", action="store_true",
                    help="if set, do not use yaw rotation augmentation")

args = parser.parse_args()


def main():
    if not args.ema:
        trainer = Trainer(args)
    else:
        if not args.mult:
            if not args.one_cycle:
                trainer = EMA_Trainer(args)
            else:
                trainer = EMA_Trainer_Onecycle(args)
        else:
            trainer = EMA_Trainer_Mult(args)
    trainer.train()
    #tester = Tester(args)
    #tester.test()


if __name__ == "__main__":
    main()
