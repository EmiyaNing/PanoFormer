from __future__ import absolute_import, division, print_function
import os
import argparse

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
#parser.add_argument("--load_weights_dir", default='./tmp_s2d3dtest/panodepth/models/weights_0', type=str)
parser.add_argument("--load_weights_dir", default=None, type=str)
parser.add_argument("--log_dir", type=str, default=os.path.join(os.path.dirname(__file__), "egformer"), help="log directory")
parser.add_argument("--log_frequency", type=int, default=1, help="number of batches between each tensorboard log")
parser.add_argument("--save_frequency", type=int, default=1, help="number of epochs between each save")
parser.add_argument("--ema_val", action="store_true", help="This parameter is used to judge whether use ema to validate")
parser.add_argument("--one_cycle", action="store_true", help="Use this to judge whether use one_cycle to determine the learning rate.")
parser.add_argument("--egformer", action="store_true", help="This parameter is used to define which model structure will be used.")

# data augmentation settings
parser.add_argument("--disable_color_augmentation", action="store_true", help="if set, do not use color augmentation")
parser.add_argument("--disable_LR_filp_augmentation", action="store_true",
                    help="if set, do not use left-right flipping augmentation")
parser.add_argument("--disable_yaw_rotation_augmentation", action="store_true",
                    help="if set, do not use yaw rotation augmentation")

args = parser.parse_args()


def main():
    trainer = EMA_Trainer_Onecycle(args)
    trainer.train()
    #tester = Tester(args)
    #tester.test()


if __name__ == "__main__":
    main()
