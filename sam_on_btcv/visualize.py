import cv2
import numpy as np

def vis_mask(mask_path):
    '''
    make mask image colorful
    '''
    mask = np.array(cv2.imread(mask_path,-1))
    vis_mask = np.zeros(list(mask.shape) + [3])
    print(np.unique(mask))
    for id in np.unique(mask):
        print((mask == id).shape)
        vis_mask[mask==id,:] = np.random.randint(0, 255, size=3)
    print(vis_mask.shape)    
    cv2.imwrite('/home/lyz/ML-SAM-Project/vis_mask.png', vis_mask)

def overlap_masks(img, masks):
    '''
    save overlaped masks into one image for visualizaion
    input:
        img:input image (H,W,3)
        masks:dict of sam output format----keys:(segmentation, area, predicted_iou)
    output: 
        colorful overlap masks in one image  (H,W,3)
    '''
    masks.sort(key=lambda x: x['area'], reverse=True)
    vis_img = np.zeros_like(img)
    for mask in masks:
        vis_img[mask['segmentation']] = np.random.randint(0, 255, size=3)
    
    return vis_img


        