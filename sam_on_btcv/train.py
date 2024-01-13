import numpy as np
import torch
import cv2
import os

from os.path import join
import tqdm
from typing import Optional, Tuple
from torch.optim import SGD, Adam

from preprocess_dataset import id_to_label
from visualize import *

from segment_anything.modeling import Sam
from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from my_pridictor import MyPredictor
from btcv_dataset import BtcvDataset, btcv_collate_fn
from criterion import dice_loss


device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def validate(val_dataloader, my_predictor:MyPredictor):
    my_predictor.model.eval()
    dice = np.zeros(13)
    organ_count = np.zeros(13)
    for img,embedding, label, prompts,batch_prompts, appearance in tqdm.tqdm(val_dataloader):
        label = label[0].to(device, non_blocking=True)
        img = img[0]
        prompts = prompts[0]
        appearance = appearance[0]
        #将13个器官的prompts组合成一个batch的prompts
        batch_prompts['multimask_output'] = False
        batch_prompts['return_logits'] = True
        
        my_predictor.set_image(img)
        masks, iou_predictions, low_res_masks = my_predictor.my_predict(**batch_prompts)
        for organ_id in range(1, 14):
            if(not appearance[organ_id-1]):
                continue
            organ_mask = masks[organ_id-1]
            organ_label = label == organ_id
            organ_dice = 1 - dice_loss(organ_mask, organ_label)
            dice[organ_id-1] += organ_dice.item()
            organ_count[organ_id-1] += 1
            
    #log dice of all organ respectively and mean dice
    per_organ_dice = dice/organ_count
    for organ_id in range(1, 14):
        print(f'dice of {id_to_label[organ_id]}: ', per_organ_dice[organ_id-1])
    print('mean dice: ', per_organ_dice.mean())


def train_one_epoch(train_dataloader, my_predictor:MyPredictor, optimizer):
    my_predictor.model.train()
    loss = None
    count = 0
    img_count = 0
    for img,embedding, label, prompts,batch_prompts, appearance in tqdm.tqdm(train_dataloader):
        label = label[0].to(device, non_blocking=True)
        img = img[0]
        prompts = prompts[0]
        appearance = appearance[0]
        #将13个器官的prompts组合成一个batch的prompts
        batch_prompts['multimask_output'] = False
        batch_prompts['return_logits'] = True
        
        my_predictor.set_image(img, image_embedding=embedding)
        masks, iou_predictions, low_res_masks = my_predictor.my_predict(**batch_prompts)
        for organ_id in range(1, 14):
            if(not appearance[organ_id-1]):
                continue
            organ_mask = masks[organ_id-1]
            organ_label = label == organ_id
            organ_loss = dice_loss(organ_mask, organ_label)
            if(loss is None):
                loss = organ_loss
            else:
                loss = loss + organ_loss
            count += 1
        #update parameters
        if((img_count+1)%10 == 0):
            loss = loss/count
            if((img_count+1)%100==0):
                print('training loss:', loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count = 0
            loss = None
        img_count += 1
            
        
def train_all():
    sam = sam_model_registry["vit_b"](checkpoint="../ckpts/sam_vit_b_01ec64.pth").cuda()
    my_predictor = MyPredictor(sam)
    
    optimizer = Adam(my_predictor.model.mask_decoder.parameters(), lr=1e-4)
    
    train_dataset = BtcvDataset(base_dir='../data/processed', split='train', prompt_class=['point'])
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=btcv_collate_fn,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )
    val_dataset = BtcvDataset(base_dir='../data/processed', split='val', prompt_class=['point'])
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        collate_fn=btcv_collate_fn,
        pin_memory=True,
        prefetch_factor=2,
        num_workers=4,
    )
    
    for epoch in range(10):
        print('epoch:', epoch)
        train_one_epoch(train_dataloader, my_predictor, optimizer)
        validate(val_dataloader,my_predictor)
        if(epoch%5 == 0):
            torch.save(my_predictor.model.mask_decoder.state_dict(), '../ckpts/mask_decoder_{}.pth'.format(epoch))
            # torch.save(my_predictor.model.image_encoder.state_dict(), '../ckpts/image_encoder_{}.pth'.format(epoch))
            # torch.save(my_predictor.model.prompt_encoder.state_dict(), '../ckpts/prompt_encoder_{}.pth'.format(epoch))
            # torch.save(optimizer.state_dict(), '../ckpts/optimizer_{}.pth'.format(epoch))

if __name__ == '__main__':
    train_all()
