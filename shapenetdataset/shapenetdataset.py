#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: dataset.py
@Time: 2020/1/2 10:26 AM
"""

import os
import torch
import json
import h5py
from glob import glob
import numpy as np
import torch.utils.data as data


shapenetpart_cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4, 
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9, 
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
shapenetpart_seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
shapenetpart_seg_start_index = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta_xz = np.pi*2 * np.random.rand()
    """
        y
        |     
        |
        |_____x
       /
      /
      z
    """
    theta_yz = np.pi*2 * np.random.rand()
    theta_xy = np.pi*2 * np.random.rand()
    rotation_matrix_horizontal_xz = np.array([[np.cos(theta_xz), -np.sin(theta_xz)],[np.sin(theta_xz), np.cos(theta_xz)]])
    rotation_matrix_horizontal_yz = np.array([[np.cos(theta_yz), np.sin(theta_yz)],[-np.sin(theta_yz), np.cos(theta_yz)]])
    rotation_matrix_horizontal_xy = np.array([[np.cos(theta_xy), np.sin(theta_xy)],[-np.sin(theta_xy), np.cos(theta_xy)]])

    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix_horizontal_xz) # random rotation (x,z)
    pointcloud[:,[1,2]] = pointcloud[:,[1,2]].dot(rotation_matrix_horizontal_yz) # random rotation (y,z)
    pointcloud[:,[0,1]] = pointcloud[:,[0,1]].dot(rotation_matrix_horizontal_xy) # random rotation (x,y)
    return pointcloud


class ShapenetDataset(data.Dataset):
    def __init__(self, root, dataset_name='modelnet40', class_choice=None,
            num_points=2048, split='train', load_name=True, load_file=True,
            segmentation=False, random_rotate=False, random_jitter=False, 
            random_translate=False, rotate_path=None, normals_path=None, rotate_path_debug=None, rotsym_normals_path=None):

        assert dataset_name.lower() in ['shapenetcorev2', 'shapenetpart', 
            'modelnet10', 'modelnet40', 'shapenetpartpart']
        assert num_points <= 2048        

        if dataset_name in ['shapenetcorev2', 'shapenetpart', 'shapenetpartpart']:
            assert split.lower() in ['train', 'test', 'val', 'trainval', 'all']
        else:
            assert split.lower() in ['train', 'test', 'all']

        if dataset_name not in ['shapenetpart'] and segmentation == True:
            raise AssertionError

        self.root = os.path.join(root, dataset_name + '_hdf5_2048')
        self.dataset_name = dataset_name
        self.class_choice = class_choice
        self.num_points = num_points
        self.split = split
        self.load_name = load_name
        self.load_file = load_file
        self.segmentation = segmentation
        self.random_rotate = random_rotate
        self.random_jitter = random_jitter
        self.random_translate = random_translate
        
        self.path_h5py_all = []
        self.path_name_all = []
        self.path_file_all = []

        if self.split in ['train', 'trainval', 'all']:   
            self.get_path('train')
        if self.dataset_name in ['shapenetcorev2', 'shapenetpart', 'shapenetpartpart']:
            if self.split in ['val', 'trainval', 'all']: 
                self.get_path('val')
        if self.split in ['test', 'all']:   
            self.get_path('test')

        data, label, seg = self.load_h5py(self.path_h5py_all)

        if self.load_name or self.class_choice != None:
            self.name = np.array(self.load_json(self.path_name_all))    # load label name

        if self.load_file:
            self.file = np.array(self.load_json(self.path_file_all))    # load file name
        
        self.data = np.concatenate(data, axis=0)
        self.label = np.concatenate(label, axis=0) 
        if self.segmentation:
            self.seg = np.concatenate(seg, axis=0) 

        if self.class_choice != None:
            indices = (self.name == class_choice)
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.name = self.name[indices]
            if self.segmentation:
                self.seg = self.seg[indices]
                id_choice = shapenetpart_cat2id[class_choice]
                self.seg_num_all = shapenetpart_seg_num[id_choice]
                self.seg_start_index = shapenetpart_seg_start_index[id_choice]
            if self.load_file:
                self.file = self.file[indices]
        elif self.segmentation:
            self.seg_num_all = 50
            self.seg_start_index = 0

        self.random_debug = np.random.rand(3, self.__len__())
        if rotate_path:
            #self.generate_rotations(rotate_path_debug)
            self.generate_rotations_onematrix(rotate_path)
            if normals_path:
                self.generate_rotation_normals(normals_path)

            if rotsym_normals_path:
                self.generate_rotation_normals_rotsym(rotsym_normals_path, def_sym=torch.tensor([0, 1.0, 0]).double())

    def get_path(self, type):
        path_h5py = os.path.join(self.root, '%s*.h5'%type)
        paths = glob(path_h5py)
        paths_sort = [os.path.join(self.root, type + str(i) + '.h5') for i in range(len(paths))]
        self.path_h5py_all += paths_sort
        if self.load_name:
            paths_json = [os.path.join(self.root, type + str(i) + '_id2name.json') for i in range(len(paths))]
            self.path_name_all += paths_json
        if self.load_file:
            paths_json = [os.path.join(self.root, type + str(i) + '_id2file.json') for i in range(len(paths))]
            self.path_file_all += paths_json
        return 

    def load_h5py(self, path):
        all_data = []
        all_label = []
        all_seg = []
        for h5_name in path:
            f = h5py.File(h5_name, 'r+')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            if self.segmentation:
                seg = f['seg'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
            if self.segmentation:
                all_seg.append(seg)
        return all_data, all_label, all_seg

    def filter_by_name(self, name):
        new_data = []
        for i, (data) in enumerate(self.data):
            if self.name[i] == "airplane":
                new_data.append(data)
        self.data = new_data

    def load_json(self, path):
        all_data = []
        for json_name in path:
            j =  open(json_name, 'r+')
            data = json.load(j)
            all_data += data
        return all_data

    def __getitem__(self, item):
        #sample first self.num_points
        #point_set = self.data[item][:self.num_points]
        #############

        #sample random subset of self.num_points
        points_in_dataset = self.data.shape[1]
        subsample_point_indexes = np.random.choice(np.arange(points_in_dataset), self.num_points, replace=False)
        point_set = torch.tensor(self.data[item][subsample_point_indexes])
        #############


        label = self.label[item]
        if self.load_name:
            name = self.name[item]  # get label name
        if self.load_file:
            file = self.file[item]  # get file name

        if self.random_rotate:
            #point_set = rotate_pointcloud(point_set)
            point_set = self.rotate_using_saved_rotations_onematrix(point_set, item)
        if self.random_jitter:
            point_set = jitter_pointcloud(point_set)
        if self.random_translate:
            point_set = translate_pointcloud(point_set)

        # convert numpy array to pytorch Tensor
        #point_set = torch.from_numpy(point_set)
        label = torch.from_numpy(np.array([label]).astype(np.int64))
        label = label.squeeze(0)
        
        if self.segmentation:
            seg = self.seg[item]
            seg = torch.from_numpy(seg)
            return point_set, label, seg, name, file
        else:
            return point_set, label, name, file

    def __len__(self):
        return self.data.shape[0]
    """
    def generate_rotations(self, path):
        self.rotationsdebug = None

        if os.path.exists(str(path)):
            print(f"Loading rotations from {str(path)}")
            self.rotationsdebug = np.load(path)
        else:
            print(f"Generating rotations at {str(path)}")
            n_objects = self.__len__()

            #theta_xz = np.pi * 2 * np.random.rand(n_objects)
            #theta_yz = np.pi * 2 * np.random.rand(n_objects)
            #theta_xy = np.pi * 2 * np.random.rand(n_objects)

            theta_xz = np.pi * 2 * self.random_debug[0]
            theta_yz = np.pi * 2 * self.random_debug[1]
            theta_xy = np.pi * 2 * self.random_debug[2]

            rotation_matrix_horizontal_xz = np.array(
                [[np.cos(theta_xz), -np.sin(theta_xz)],
                 [np.sin(theta_xz), np.cos(theta_xz)]])
            rotation_matrix_horizontal_yz = np.array(
                [[np.cos(theta_yz), np.sin(theta_yz)], [-np.sin(theta_yz), np.cos(theta_yz)]])
            rotation_matrix_horizontal_xy = np.array(
                [[np.cos(theta_xy), np.sin(theta_xy)], [-np.sin(theta_xy), np.cos(theta_xy)]])

            self.rotationsdebug = np.array([rotation_matrix_horizontal_xz,rotation_matrix_horizontal_yz,rotation_matrix_horizontal_xy])
            np.save(path, self.rotationsdebug)
    """
    def generate_rotations_onematrix(self, path):
        self.rotations = None

        if os.path.exists(str(path)):
            print(f"Loading rotations from {str(path)}")
            self.rotations = torch.tensor(np.load(path))
        else:
            print(f"Generating rotations at {str(path)}")
            n_objects = self.__len__()

            theta_xz = np.pi * 2 * self.random_debug[0]
            theta_yz = np.pi * 2 * self.random_debug[1]
            theta_xy = np.pi * 2 * self.random_debug[2]

            array0 = np.zeros(n_objects)
            array1 = array0 + 1

            rotation_matrix_horizontal_xz = np.array(
                [[np.cos(theta_xz), array0, -np.sin(theta_xz)],
                 [array0, array1, array0],
                 [np.sin(theta_xz), array0, np.cos(theta_xz)]])
            rotation_matrix_horizontal_yz = np.array(
                [[array1, array0, array0],
                 [array0, np.cos(theta_yz), np.sin(theta_yz)],
                 [array0, -np.sin(theta_yz), np.cos(theta_yz)]])
            rotation_matrix_horizontal_xy = np.array(
                [[np.cos(theta_xy), np.sin(theta_xy), array0],
                 [-np.sin(theta_xy), np.cos(theta_xy), array0],
                 [array0, array0, array1]
                 ])

            rot_multiplied1 = np.einsum("ijb,jkb->ikb", rotation_matrix_horizontal_xz, rotation_matrix_horizontal_yz)
            rot_multiplied2 = np.einsum("ijb,jkb->ikb", rot_multiplied1, rotation_matrix_horizontal_xy)

            self.rotations = torch.tensor(rot_multiplied2)

            np.save(path, self.rotations)

    """
    def rotate_using_saved_rotations(self, pointcloud, item):
        rotation_matrix_horizontal_xz = self.rotations[0]
        rotation_matrix_horizontal_yz = self.rotations[1]
        rotation_matrix_horizontal_xy = self.rotations[2]

        pointcloud[:, [0, 2]] = pointcloud[:, [0, 2]].dot(rotation_matrix_horizontal_xz[:, :, item])  # random rotation (x,z)
        pointcloud[:, [1, 2]] = pointcloud[:, [1, 2]].dot(rotation_matrix_horizontal_yz[:, :, item])  # random rotation (y,z)
        pointcloud[:, [0, 1]] = pointcloud[:, [0, 1]].dot(rotation_matrix_horizontal_xy[:, :, item])  # random rotation (x,y)

        return pointcloud
    """
    def rotate_using_saved_rotations_onematrix(self, pointcloud, item):
        rotation_matrix = self.rotations
        return pointcloud @ rotation_matrix[:, :, item].float()

    def generate_rotation_normals(self, normals_path, def_sym=torch.tensor([1.0, 0, 0]).double()):
        rotation_matrix_onematrix = self.rotations
        default_symmetry = def_sym
        inverse_of_transpose = torch.linalg.inv(torch.tensor(rotation_matrix_onematrix).transpose(0,2))
        self.rot_normals =  default_symmetry @ inverse_of_transpose
        np.save(normals_path, self.rot_normals)
        np.savetxt(str(normals_path) + ".txt", self.rot_normals)
        return self.rot_normals

    def generate_rotation_normals_rotsym(self, normals_path, def_sym=torch.tensor([0, 1.0, 0]).double()):
        rotation_matrix_onematrix = self.rotations
        default_symmetry = def_sym
        inverse_of_transpose = torch.linalg.inv(torch.tensor(rotation_matrix_onematrix).transpose(0,2))
        self.rot_normals_rotsym =  default_symmetry @ inverse_of_transpose
        np.save(normals_path, self.rot_normals_rotsym)
        np.savetxt(str(normals_path) + ".txt", self.rot_normals_rotsym)
        return self.rot_normals_rotsym

if __name__ == '__main__':
    root = os.getcwd()

    # choose dataset name from 'shapenetcorev2', 'shapenetpart', 'modelnet40' and 'modelnet10'
    dataset_name = 'shapenetcorev2'

    # choose split type from 'train', 'test', 'all', 'trainval' and 'val'
    # only shapenetcorev2 and shapenetpart dataset support 'trainval' and 'val'
    split = 'train'

    d = ShapenetDataset(root=root, dataset_name=dataset_name, num_points=2048, split=split)
    print("datasize:", d.__len__())

    item = 0
    ps, lb, n, f = d[item]
    print(ps.size(), ps.type(), lb.size(), lb.type(), n, f)

    for i in range(len(d)):
        if d.name[i] == "lamp":
            print(i)