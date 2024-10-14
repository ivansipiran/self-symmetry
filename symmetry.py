from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

from dgcnn.model import DGCNN_cls_orig, DGCNN_cls, DGCNN_cls3vec
from log import CDSLossLogger

try:
    import open3d
except:
    pass
import torch
import options as options
import util
import torch.optim as optim
from losses import chamfer_distance
from data_handler import get_dataset
from torch.utils.data import DataLoader
from models import SymmetryNetwork
from torch.nn import L1Loss
import sys
import os
import polyscope as ps
from pathlib import Path
import traceback
import numpy as np
import matplotlib.pyplot as plt
from Pointnet.models import pointnet2_cls_ssg
from torch.utils.data import Subset
from models3vector import SymmetryNetwork2 as SymmetryNetwork3Vector
from models3vector import SymmetryNetwork14V as SymmetryNetwork14Vector

import sys


def chamfer_detm1_symtranspose(original: torch.Tensor, transformed: torch.Tensor, sym_matrix, distance = chamfer_distance, mse = False):
    loss1 = distance(original, transformed, mse)
    R = sym_matrix
    loss2 = torch.pow(torch.det(R) + 1,2).sum() # batches sum()
    RtR = torch.matmul(torch.transpose(R, 2, 1), R)
    I = torch.eye(R.size(-1), dtype=R.dtype, device=R.device)
    loss3 = torch.mean(torch.abs(RtR - I), dim=[0, 1, 2])
    loss = loss1 + loss2 + loss3
    return loss


def sel_object(args, device):
    if args.dataset == "oneobject":
        target_pc: torch.Tensor = util.get_input(args, center=True).unsqueeze(0).permute(0, 2, 1).to(device)
        if 0 < args.max_points < target_pc.shape[2]:
            indx = torch.randperm(target_pc.shape[2])
            target_pc = target_pc[:, :, indx[:args.cut_points]]
        return target_pc
    elif args.dataset == "shapenet":
        return sel_dataloader(args, device)[args.shapenetitem][0][None, :, :].transpose(1, 2)

def sel_dataloader(args, device):
    if args.dataset == "oneobject":
        target_pc = sel_object(args, device)
        data_loader = get_dataset(args.sampling_mode)(target_pc[0].transpose(0, 1), device, args)

    elif args.dataset == "shapenet":
        root = str(args.pc)
        # choose dataset name from 'shapenetcorev2', 'shapenetpart', 'modelnet40' and 'modelnet10'
        dataset_name = 'shapenetcorev2'
        # choose split type from 'train', 'test', 'all', 'trainval' and 'val'
        # only shapenetcorev2 and shapenetpart dataset support 'trainval' and 'val'
        split = args.shapenet_split
        data_loader = get_dataset("shapenet")(root=root, dataset_name=dataset_name, num_points=2048,
                                                      split=split, random_rotate=args.random_rotate,
                                              rotate_path=args.rotate_path)
        if args.shapenetsubset != "all":
            airplanes = []
            for i, (d2, label, name, file) in enumerate(data_loader):
                if name == args.shapenetsubset:
                    airplanes.append(i)
            data_loader = Subset(data_loader, airplanes)

    """
        elif args.dataset == "shapenetclasssubset":
        args.dataset = "shapenet"
        data_loader = sel_dataloader(args, device)
        args.dataset = "shapenetclasssubset"

        airplanes = []
        for i, (d2, label, name, file) in enumerate(data_loader):
            if name == "airplane":
                airplanes.append(i)
        data_loader = Subset(data_loader, airplanes)
    """
    return data_loader



def unpack_elements(args, train_loader_output):
    output = {}
    if args.dataset2 == "oneobject":
        output["points"] = train_loader_output[1]
        output["label"] = "None"
        output["name"] = "None"
        output["file"] = "None"
        return output
    elif args.dataset2 == "shapenet":
        output["points"] = train_loader_output[0].transpose(2, 1)
        output["label"] = train_loader_output[1]
        output["name"] = train_loader_output[2]
        output["file"] = train_loader_output[3]
        return output

def sel_optimizer(model_parameters, args):
    if args.optimizer == "Adam":
        return optim.Adam(model_parameters, lr=args.lr)
    elif args.optimizer == "AdamW":
        return optim.AdamW(model_parameters, lr=args.lr)

    raise RuntimeError


def sel_model(args, num_class, device=None):
    if args.model == "PointNet1":
        return SymmetryNetwork(num_class=num_class)
    if args.model == "PointNet2":
        return pointnet2_cls_ssg.get_model(num_class=num_class, normal_channel=False)
    if args.model == "Dgcnn":
        return DGCNN_cls(args, output_channels=num_class)
    if args.model == "Dgcnn_batches":
        return DGCNN_cls_orig(args, output_channels=num_class)
    if args.model == "Pvcnn":
        from pvcnn.models.shapenet import PVCNN
        return PVCNN(num_classes=num_class, extra_feature_channels=0, num_shapes=0)
    if args.model == "PointNet_3vector":
        return SymmetryNetwork3Vector()
    if args.model == "PointNet2_3vector":
        return pointnet2_cls_ssg.get_model3vector(normal_channel=False)
    if args.model == "Dgcnn_3vector":
        return DGCNN_cls3vec(args, output_channels=9)
    if args.model == "PointNet_14vector":
        return SymmetryNetwork14Vector(device)

    raise RuntimeError


def sel_scheduler(optimizer, args):
    if args.scheduler == "CosineLR":
        return CosineAnnealingLR(optimizer, T_max = args.iterations, eta_min=0.00001)
    elif args.scheduler == "CosineRestarts":
        return CosineAnnealingWarmRestarts(optimizer, T_0 = int(args.iterations/args.scheduler_restart), eta_min = 0.001)
    elif args.scheduler is None:
        return util.NoScheduler()

