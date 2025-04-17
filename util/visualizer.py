import numpy as np
import os
import sys
import ntpath
import time
import torch
import torchvision
import matplotlib.pyplot as plt
import math
from PIL import Image
from . import util
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_pil_image

def save_images(opt,visuals, img_path):
    import ipdb
    os.makedirs(opt.results_dir, exist_ok=True)
    dict_length = len(visuals)
    idx = img_path[0].split(".", -1)[-2].split("/")[-1]
    for label, img in visuals.items():
        img_file_name = idx.zfill(4) + '.png'
        save_img = (img + 1)/2
        save_img = torch.squeeze(save_img)
        save_img = to_pil_image(save_img)
        os.makedirs(os.path.join(opt.results_dir, label), exist_ok=True)
        save_img.save(os.path.join(opt.results_dir, label, img_file_name), 'png')

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False
        self.current_epoch = 0
        self.ncols = opt.display_ncols
        self.logdir = opt.logdir
        self.writer = SummaryWriter(self.logdir)

    def display_current_results(self, visuals, current_iter):
        for label, image in visuals.items():
            image = (image + 1 ) / 2
            image = torch.squeeze(image)
            self.writer.add_image(label, image, global_step=current_iter, dataformats='CHW')


    def plot_current_losses(self, epoch, counter_ratio, losses):
        idx = 0
        for label, loss in losses.items():
            self.writer.add_scalar(label, loss, counter_ratio + epoch)

    def reset(self):
        pass
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
