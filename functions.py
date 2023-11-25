import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.vision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
paddle.seed(42)
np.random.seed(42)

import random

def minmax(data, smooth=1e-7):
    mean = np.mean(data)
    da_max=np.max(data)
    da_min=np.min(data)
    return (data-mean+smooth)/(da_max-da_min+smooth)

def standardization(data,smooth=1e-7):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean + smooth) / (std + smooth)

def gammaadjust(img, out_range=[-1, 1], gamma=1):
    low_in, high_in = np.min(img), np.max(img)
    low_out, high_out = out_range[0], out_range[1]
    img_out = (((img - low_in) / (high_in - low_in)) ** gamma) * (high_out - low_out) + low_out
    return img_out

def SDA_Augment(image):
    xxx=random.random()
    if xxx<0.5:
        sigma = np.random.uniform(0.3, 0.8)
        image=gammaadjust(image,gamma=sigma)
    if xxx>=0.5:
        sigma = np.random.uniform(1.3, 2.7)
        image=gammaadjust(image,gamma=sigma)
    if random.random()<0.6:
        blur=np.random.choice(np.array([3,5,7,9,11]))
        image=cv2.GaussianBlur(image,ksize=(blur,blur),sigmaX=0.)
    return np.squeeze(image)

def sda_pick_data(path,num_slices=0,with_label=0):
    images=[]
    labels=[]
    sda1_images=[]
    #sda2_images=[]

    filenames=os.listdir(path)
    filenames.sort()

    count=0
    while(count<num_slices):
        i=np.random.randint(low=0,high=len(filenames))
        img = np.load(path.replace('gt_','')+'{}'.format(filenames[i].replace('_gt','')))

        if( np.unique(img).shape[0] < 30000 ):
            continue

        count+=1

        if(with_label==1):
            lal = np.load(path+'{}'.format(filenames[i]))
            labels.append(lal)

        sda1_img=SDA_Augment(np.expand_dims(img.copy(),axis=-1))
        #sda2_img=SDA_Augment(np.expand_dims(img.copy(),axis=-1))

        images.append(np.expand_dims(img,axis=0))
        sda1_images.append(np.expand_dims(sda1_img,axis=0))
        #sda2_images.append(np.expand_dims(sda2_img,axis=0))

    return paddle.to_tensor(images).astype('float32'),paddle.to_tensor(sda1_images).astype('float32'),paddle.to_tensor(labels).astype('int32')

def pick_data(path,training,num_slices=0,num_foreground_slices=0,start=0,end=0):
    images=[]
    labels=[]
    filenames=os.listdir(path)
    filenames.sort()
    dic_all = [[],[]]
    sda1_images=[]
    if(training==1):
        count=0
        while(count<num_slices):
            i=np.random.randint(low=0,high=len(filenames))
            img = np.load(path.replace('gt_','')+'{}'.format(filenames[i].replace('_gt','')))
            lal = np.load(path+'{}'.format(filenames[i]))
            if(count>=num_foreground_slices):
                piner=img
                if( np.unique(piner).shape[0] < 30000 ):
                    continue
            if(count<num_foreground_slices):
                piner=lal
                if( np.max(piner) == np.min(piner) ):
                    continue
            count+=1
            dic_temp0 = []
            dic_temp1 = []
            #sda1_img=SDA_Augment(np.expand_dims(img.copy(),axis=-1))

            images.append(np.expand_dims(img,axis=0))
            #sda1_images.append(np.expand_dims(sda1_img,axis=0))
            labels.append(lal)
            
              
 
    
    return paddle.to_tensor(images).astype('float32'),paddle.to_tensor(labels).astype('int32'),dic_all

def plot():
    with open('log.txt',mode='r') as f:
        res=f.readlines()

    plt_loss_gen=[]
    plt_loss_con_seg=[]
    plt_loss_con_segcon=[]
    plt_loss_consis=[]
    plt_loss_dis=[]
    plt_loss_iden=[]
    plt_loss_seg=[]

    for i in range(0,len(res),1):
        if(res[i].startswith('epoch')==0):
            continue

        plt_loss_gen.append(res[i].split('generator:')[1].split(' ')[0])
        plt_loss_con_seg.append(res[i].split('contrastive_seg:')[1].split(' ')[0])
        plt_loss_con_segcon.append(res[i].split('contrastive_segcon:')[1].split(' ')[0])
        plt_loss_consis.append(res[i].split('consistency:')[1].split(' ')[0])
        plt_loss_iden.append(res[i].split('identity:')[1].split(' ')[0])
        plt_loss_dis.append(res[i].split('discriminator:')[1].split(' ')[0])
        plt_loss_seg.append(res[i].split('segment:')[1].split(' ')[0])

    plt_loss_gen=np.array(plt_loss_gen).astype('float32')
    plt_loss_con_seg=np.array(plt_loss_con_seg).astype('float32')
    plt_loss_con_segcon=np.array(plt_loss_con_segcon).astype('float32')
    plt_loss_consis=np.array(plt_loss_consis).astype('float32')
    plt_loss_iden=np.array(plt_loss_iden).astype('float32')
    plt_loss_dis=np.array(plt_loss_dis).astype('float32')
    plt_loss_seg=np.array(plt_loss_seg).astype('float32')


    plt.xlabel('epoch')
    plt.plot(plt_loss_gen,label='loss_gen')
    plt.plot(plt_loss_con_seg,label='loss_con_seg')
    plt.plot(plt_loss_con_segcon,label='loss_con_segcon')
    plt.plot(plt_loss_consis,label='loss_consistency')
    #plt.plot(plt_loss_iden,label='loss_iden')
    plt.plot(plt_loss_dis,label='loss_dis')
    #plt.plot(plt_loss_seg,label='loss_seg')

    plt.legend()
    plt.show()


