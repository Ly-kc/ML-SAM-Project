import torch
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import os
from os.path import join
import cv2
import numpy as np


class BtcvDataset(Dataset): #继承Dataset
    def __init__(self, base_dir, split='train',prompt_class=['point']): #__init__是初始化该类的一些基础参数
        self.split = split
        if(split in ['train','val']):
            self.data_dir = join(base_dir, 'Training', 'All')
        else:
            self.data_dir = join(base_dir, 'Testing', 'All')
        img_names = os.listdir(self.data_dir)
        img_names = self.filter_img(img_names)
        self.img_name_list = img_names
        self.prompt_class = prompt_class
    
    def filter_img(self, img_names):
        filtered_img_names = []
        for img_name in img_names:
            if('img' not in img_name):
                continue
            img_num = int(img_name[3:7])
            if((self.split == 'train' and img_num >= 35) or self.split == 'val' and img_num < 35):
                continue
            filtered_img_names.append(img_name)
        return filtered_img_names
                
    def __len__(self):#返回整个数据集图片的数目
        return len(self.img_name_list)
    
    def __getitem__(self,index):#根据索引index返回dataset[index]
        '''
        :param index: (int) 索引,代表将要取出list中第几张图片
        :return img: np.ndarray(H,W);取出的灰度图
        :return label: np.ndarray(H,W);取出的图片的label
        :return prompts: list[dict{},dict{}...].每个dict代表图片中一个器官的prompt,dict的格式参考criterion.py的第76行
        :return appearance: np.ndarray(13,dtype=bool). 图片中是否存在每个器官的gt
        '''
        img_path = join(self.data_dir, self.img_name_list[index])
        label_path = img_path.replace('img','mask')
        img = np.array(cv2.imread(img_path))
        label = np.array(cv2.imread(label_path,-1))

        appearance = np.ones(13,dtype=bool)
        prompts = []

        for organ_id in range(1, 14):
            prompt = {}
            mask = label == organ_id
            if('point' in self.prompt_class):
                sample_num = 2                
                #sample points randomly in gt area
                pixels = mask.nonzero()
                if(mask.nonzero()[0].shape[0] == 0): #如果gt中不存在该器官
                    appearance[organ_id-1] = False
                    rand_point = [[0,0] for i in range(sample_num)]
                else:  
                    rand_points = np.random.randint(low=0,high=pixels[0].shape[0],size=sample_num)
                    rand_point = [[pixels[0][rand_points[i]],pixels[1][rand_points[i]]] for i in range(sample_num)]
                #set parameter fot predict
                prompt['point_coords'] = np.array(rand_point) 
                prompt['point_labels'] = np.ones(sample_num)
                # print(prompts['point_coords'])
            if('box' in self.prompt_class):
                pixels = mask.nonzero()
                if(mask.nonzero()[0].shape[0] == 0):
                    appearance[organ_id-1] = False
                    box = np.array([0,0,0,0])
                else:
                    x1 = pixels[0].min()
                    x2 = pixels[0].max()
                    y1 = pixels[1].min()
                    y2 = pixels[1].max()
                box = np.array([x1, y1, x2, y2])
                prompt['box'] = box
            
            prompts.append(prompt)

        return img, label, prompts, appearance

        
def btcv_collate_fn(batch):
    '''
    :param batch: list[img,label,prompts,appearance]
    :return img: np.ndarray(B,H,W,C)
    :return label: torch.Tensor(B,H,W)
    :return prompts: list[dict{},dict{}...].每个dict代表图片中一个器官的prompt,dict的格式参考criterion.py的第76行
    :return appearance: torch.Tensor(B,13,dtype=bool). 图片中是否存在每个器官的gt
    '''
    img, label, prompts, appearance = zip(*batch)
    img = np.stack(img, axis=0)
    label = torch.from_numpy(np.stack(label, axis=0))
    appearance = np.stack(appearance, axis=0)
    return img, label, prompts, appearance