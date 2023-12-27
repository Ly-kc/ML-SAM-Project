import torch
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import os
from os.path import join
import cv2
import numpy as np


class BtcvData(Dataset): #继承Dataset
    def __init__(self, base_dir, split='Training',prompt_class=['point']): #__init__是初始化该类的一些基础参数
        self.data_dir = join(base_dir, split, 'All')
        # self.img_name_list = All中所有img的name
        img_names = os.listdir(self.data_dir)
        img_names = [img_name for img_name in img_names if 'img' in img_name]
        self.img_name_list = img_names
        self.prompt_class = prompt_class
        
    
    def __len__(self):#返回整个数据集图片的数目
        # raise NotImplementedError
        return len(self.img_name_list)
    
    def __getitem__(self,index):#根据索引index返回dataset[index]
        '''
        :param index: (int) 索引,代表将要取出list中第几张图片
        :return img: np.ndarray(H,W);取出的灰度图
        :return label: np.ndarray(H,W);取出的图片的label
        :return prompts: list[dict{},dict{}...].每个dict代表图片中一个器官的prompt,dict的格式参考criterion.py的第76行
        '''
        img_path = join(self.data_dir, self.img_name_list[index])
        label_path = img_path.replace('img','mask')
        img = np.array(cv2.imread(img_path,-1))
        label = np.array(cv2.imread(label_path,-1))

        prompts = []

        for organ_id in range(1, 14):
            prompt = {}
            prompt['multimask_output'] = True
            if('point' in self.prompt_class):
                sample_num = 2
                mask = label == organ_id
                #sample points randomly in gt area
                pixels = mask.nonzero()
                rand_points = np.random.randint(low=0,high=pixels[0].shape[0],size=sample_num)
                rand_point = [[pixels[0][rand_points[i]],pixels[1][rand_points[i]]] for i in range(sample_num)]
                #set parameter fot predict
                prompt['point_coords'] = np.array(rand_point) 
                prompt['point_labels'] = np.ones(sample_num)
                # print(prompts['point_coords'])
            if('box' in self.prompt_class):
                mask = label == organ_id
                pixels = mask.nonzero()
                x1 = pixels[0].min()
                x2 = pixels[0].max()
                y1 = pixels[1].min()
                y2 = pixels[1].max()
                box = np.array([x1, y1, x2, y2])
                prompt['box'] = box
            prompts.append(prompt)

        return img, label, prompts

        
        
        
        

dataloader = DataLoader(BtcvData, batch_size=1, shuffle=True, num_workers=0) #使用DataLoader加载数据