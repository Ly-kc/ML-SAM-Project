import torch
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import os
import numpy as np


class BtcvData(Dataset): #继承Dataset
    def __init__(self, base_dir, split='Training',prompt_class=['point']): #__init__是初始化该类的一些基础参数
        dayta_dir = join(base_dir, split, 'All')
        self.img_name_list = All中所有img的name
        
    
    def __len__(self):#返回整个数据集图片的数目
        raise NotImplementedError
    
    def __getitem__(self,index):#根据索引index返回dataset[index]
        '''
        :param index: (int) 索引,代表将要取出list中第几张图片
        :return img: np.ndarray(H,W);取出的灰度图
        :return label: np.ndarray(H,W);取出的图片的label
        :return prompts: list[dict{},dict{}...].每个dict代表图片中一个器官的prompt,dict的格式参考criterion.py的第76行
        '''
        
        
        
        

dataloader = DataLoader(BtcvData, batch_size=1, shuffle=True, num_workers=0) #使用DataLoader加载数据