import nnc

import sys
sys.settrace

from ultralytics import YOLO
import torch
import deepCABAC
import torchvision.models as models
import numpy as np
from tqdm import tqdm

import os
from time import perf_counter

def inceptionv3_save ():
    model = models.inception_v3(weights='DEFAULT')
    path = "/home/jiovana/Documents/nncodec/example/models/inceptionv3_full.pth"
    torch.save(model,path)

def alexnet_save ():
    model = models.alexnet(weights='DEFAULT')
    path = "/home/jiovana/Documents/nncodec/example/models/alexnet_full.pth"
    torch.save(model,path)

def deeplabv3_save():
    model = models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
    path = "/home/jiovana/Documents/nncodec/example/models/deeplabv3_full.pth"
    torch.save(model,path)

def yolov5_save ():
    path1 = "/home/jiovana/Documents/nncodec/example/models/yolo11s.pt"
    model = YOLO(path1)
    #model = torch.hub.load('ultralytics/yolov5','yolov5s', pretrained=True)
    #path = "/home/jiovana/Documents/nncodec/example/models/yolov5_full.pth"
    torch.save(model,"/home/jiovana/Documents/nncodec/example/models/yolo11s_full.pt")

def unet_save ():
    model = torch.hub.load('milesial/Pytorch-UNet','unet_carvana', pretrained=False, scale=0.5)
    state_dict = torch.hub.load_state_dict_from_url('https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth',  map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    path = "/home/jiovana/Documents/nncodec/example/models/unet_full.pth"
    torch.save(model,path) 

def ssd_save ():
    model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub','nvidia_ssd', weights=None, pretrained=False).to('cpu')
    location = '/home/jiovana/Downloads/pytorch_nn_models/nvidia_ssd.pth'
    state_dict = torch.load(location, map_location=torch.device('cpu'),)
    model.load_state_dict(state_dict)
    path = "/home/jiovana/Documents/nncodec/example/models/ssd_full.pth"
    torch.save(model,path) 

def bmshj2018_save ():
    from compressai.zoo import bmshj2018_factorized
    model = bmshj2018_factorized(quality=2, pretrained=True)
    path = "/home/jiovana/Documents/nncodec/example/models/bmshj_full.pth"
    torch.save(model,path)

def cheng2020_save ():
    from compressai.zoo import cheng2020_anchor
    model = cheng2020_anchor(quality=2, pretrained=True)
    path = "/home/jiovana/Documents/nncodec/example/models/cheng_full.pth"
    torch.save(model,path)

def main ():
    yolov5_save()
    model_path = '/home/jiovana/Documents/nncodec/example/models/yolo11s_full.pt'
    bitstream_path = 'bitstream_yolo11s.nnc'
    reconstructed = '/home/jiovana/Documents/nncodec/example/models/yolo11s_rec.pt'

    nnc.compress_model(model_path, bitstream_path=bitstream_path)

    nnc.decompress_model(bitstream_path, model_path=reconstructed)

    model = YOLO("/home/jiovana/Documents/nncodec/example/models/yolo11s.pt",task='detect')
    model.load_state_dict(reconstructed)
    model.save("/home/jiovana/Documents/nncodec/example/models/yolo11s_rec.pt")
    #torch.save(model, "/home/jiovana/Documents/nncodec/example/models/yolo11s_rec.pt")






if __name__ == '__main__':
    main()