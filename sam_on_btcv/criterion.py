import numpy as np
import torch
import cv2
import os
from os.path import join
import tqdm

from sam_on_btcv.preprocess_dataset import id_to_label
from visualize import *

from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

#sam的输出：dict_keyes(['segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box'])
#我们的gt:与图片相同形状的label image，元素值是0-13的整数

def dice_loss(pred, gt):
    '''
    简单定义的dice loss
    pred: (batch_size, H, W)
    gt: (H, W)
    '''
    ep = 1e-8
    intersection = torch.sum(pred * gt) + ep  # (batch_size,)
    union = torch.sum(pred, dim=(-2,-1)) + torch.sum(gt) + ep  # (batch_size,)
    loss = (1 - 2*intersection / union).mean()  #(,)
    return loss

def test(organ_id=8, gt_base='../data/processed/Training', pred_base = '../results/train'):
    '''
    对某一类器官分割后保存的结果计算dice loss
    '''
    organ_name = id_to_label[organ_id]
    pred_dir = join(pred_base, organ_name)
    gt_dir = join(gt_base,  organ_name)
    pred_names = sorted(os.listdir(pred_dir))
    
    loss = 0
    for pred_name in tqdm.tqdm(pred_names):
        pred_path = join(pred_dir, pred_name)
        gt_name = pred_name.replace('img','mask')
        gt_path = join(gt_dir, gt_name)
        pred = torch.from_numpy(np.array(cv2.imread(pred_path,-1)))
        gt = torch.from_numpy(np.array(cv2.imread(gt_path,-1)))
        loss += dice_loss(pred, gt)

    print(loss/len(pred_names))    
    

def generate_seg(sam=None, organ_id=8, prompt_class=['point'], base_dir = '../data/processed/Training/All', save_base = '../results/train'):
    '''
    对某一类器官的所有图片进行分割，保存结果
    '''
    if(sam is None):
        sam = sam_model_registry["vit_b"](checkpoint="../ckpts/sam_vit_b_01ec64.pth").cuda()
    predictor = SamPredictor(sam)
    
    organ_name = id_to_label[organ_id]
    save_dir = join(save_base, organ_name)
    os.makedirs(save_dir, exist_ok=True)
    
    img_names = os.listdir(base_dir)
    img_names = [img_name for img_name in img_names if 'img' in img_name]
    for img_name in tqdm.tqdm(img_names):
        img_path = join(base_dir, img_name)
        label_path = img_path.replace('img','mask')
        img = np.repeat(np.array(cv2.imread(img_path,-1))[...,None], 3, axis=-1)
        labels = np.array(cv2.imread(label_path,-1))
        
        semantic_mask = labels == organ_id
        if(semantic_mask.sum() == 0):
            continue
        
        predictor.set_image(img)
            
        prompts = {}
        prompts['multimask_output'] = True
        if('point' in prompt_class):
            sample_num = 2
            mask = labels == organ_id
            #sample points randomly in gt area
            pixels = mask.nonzero()
            rand_points = np.random.randint(low=0,high=pixels[0].shape[0],size=sample_num)
            rand_point = [[pixels[0][rand_points[i]],pixels[1][rand_points[i]]] for i in range(sample_num)]
            #set parameter fot predict
            prompts['point_coords'] = np.array(rand_point) 
            prompts['point_labels'] = np.ones(sample_num)
            # print(prompts['point_coords'])
        
        if('box' in prompt_class):
            raise NotImplementedError
        
        
        
        
        masks, scores, logits = predictor.predict(
            **prompts,
        )
        
        
        #save the most condifent mask only
        conf_id = np.argmax(scores)
        res_mask = masks[conf_id].astype('uint8')*255
        
        cv2.imwrite(join(save_dir, img_name), res_mask)
        

if __name__ == '__main__':
    # generate_seg()    
    test()
