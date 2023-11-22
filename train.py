import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import paddle.vision.transforms as T
import math
import os
import sys 

import train_networks
import medpy.metric.binary as mmb
import networks.unet as unet
import functions as functions

nn.initializer.set_global_initializer(nn.initializer.KaimingNormal(), nn.initializer.Constant())
paddle.seed(42)
np.random.seed(42)

segmentor=unet.UNet(num_channels=64)

source_domain_name='mr'
if(source_domain_name=='ct'):
    source_name='ct'
    target_name='mr'

elif(source_domain_name=='mr'):
    source_name='mr'
    target_name='ct'

train_networks.train(100,lr=0.0001)
paddle.save(segmentor.state_dict(),'segmentor_sup{}'.format(source_domain_name))
