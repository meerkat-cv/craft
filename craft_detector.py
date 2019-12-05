import os
import time
import logging

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from .imgproc import resize_aspect_ratio, normalizeMeanVariance, cvt2HeatmapImg, loadImageFromOpenCV
from .craft_utils import getDetBoxes, adjustResultCoordinates, adjustResultCoordinates
from .craft import CRAFT

from collections import OrderedDict


def _copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

class CraftDetector:

    def __init__(self, model_path, use_gpu=False):
        self.using_gpu = use_gpu
        self.net = CRAFT()

        if self.using_gpu:
            self.net.load_state_dict(_copyStateDict(torch.load(model_path)))
            self.net = self.net.cuda()
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = False
        else:
            self.net.load_state_dict(_copyStateDict(torch.load(
                model_path, map_location='cpu')))

        self.net.eval()


    def detect(self, image, text_threshold=0.7, link_threshold=0.4, low_text=0.4, poly=False, refine_net=None):
        with torch.no_grad():
            t0 = time.time()
            image = loadImageFromOpenCV(image)

            # resize
            mag_ratio = 50
            canvas_size = 2100
            # if self.using_gpu:
            # mag_ratio = 50.0
            # canvas_size = 2800
            # else:
            #     canvas_size = 896
            img_resized, target_ratio, size_heatmap = resize_aspect_ratio(
                image, canvas_size, interpolation=cv2.INTER_AREA, mag_ratio=mag_ratio)
            ratio_h = ratio_w = 1 / target_ratio

            # preprocessing
            x = normalizeMeanVariance(img_resized)
            x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
            x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
            if self.using_gpu:
                x = x.cuda()

            # forward pass
            y, feature = self.net(x)

            # make score and link map
            score_text = y[0, :, :, 0].cpu().data.numpy()
            score_link = y[0, :, :, 1].cpu().data.numpy()

            # TODO: Gustavo, this refine link was done for the paper, should
            # not be used.
            # if refine_net is not None:
            #     y_refiner = refine_net(y, feature)
            #     score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

            t0 = time.time() - t0
            t1 = time.time()

            # Post-processing
            boxes, polys = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

            # coordinate adjustment
            boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
            polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
            for k in range(len(polys)):
                if polys[k] is None:
                    polys[k] = boxes[k]

            t1 = time.time() - t1

            logging.debug("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

            return boxes, polys #, ret_score_text
