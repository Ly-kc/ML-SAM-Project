from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import numpy as np
import cv2

from visualize import *
        
def automask(sam, img_path):
    sam = sam_model_registry["vit_b"](checkpoint="../ckpts/sam_vit_b_01ec64.pth").cuda()
    mask_generator = SamAutomaticMaskGenerator(sam)
    img = np.array(cv2.imread('/home/lyz/ML-SAM-Project/data/processed/Training/img_gray/All/img0001_0.png'))

    masks = mask_generator.generate(img)
    vis_img = overlap_masks(img, masks)

    cv2.imwrite('/home/lyz/ML-SAM-Project/vis_img.png', vis_img)
    
    