'''
功能：KITTI数据集标签格式转换
'''

import sys
from pathlib import Path
import json
import numpy as np
import cv2
from tqdm import tqdm
sys.path.append('./src')


DATA_PATH = './data/kitti/'
import os
import math
from lib.utils.ddd_utils import compute_box_3d, project_to_image

'''
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
'''


def _bbox_to_coco_bbox(bbox):
    return [(bbox[0]), (bbox[1]),
            (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]


def read_clib(calib_path):
    f = open(calib_path, 'r')
    for i, line in enumerate(f):
        if i == 2:
            calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
            calib = calib.reshape(3, 4)
            return calib

def main():
    cats = ['Car', 'Pedestrian', 'Cyclist', 'Van', 'Truck', 'Person_sitting',
            'Tram', 'Misc', 'DontCare']
    det_cats=['Car', 'Pedestrian', 'Cyclist']

    cat_ids = {cat: i + 1 for i, cat in enumerate(cats)}

    cat_info = []
    for i, cat in enumerate(cats):
        cat_info.append({'name': cat, 'id': i + 1})

    image_set_path = Path(DATA_PATH) / 'image'
    ann_dir = Path(DATA_PATH) / 'label'
    calib_dir = Path(DATA_PATH) / 'calib'
    splits = ['train','val']

    for split in splits:
        ret = {'images': [], 'annotations': [], "categories": cat_info}
        image_set = open(DATA_PATH + '{}.txt'.format(split), 'r')
        
        for img_name in tqdm(image_set):
            if img_name[-1] == '\n':
                img_name = img_name[:-1]
            image_id = int(img_name)
            calib_path = calib_dir / f'{img_name}.txt'

            
            calib = read_clib(calib_path)
            
            image_info = {'file_name': f'{img_name}.png', 'id': int(image_id), 'calib': calib.tolist()}
        
            ret['images'].append(image_info)

            ann_path = ann_dir / f'{img_name}.txt'

            anns = open(ann_path, 'r')
            for ann in anns:
                ann = ann[:-1].split(' ')

                if ann[0] in det_cats:
                    
                    cat_id = cat_ids[ann[0]]    # 类别ID
                    alpha = float(ann[3])       
                    dim = [float(ann[8]), float(ann[9]), float(ann[10])]   # 高、宽、长
                    location = [float(ann[11]) + 0.06, float(ann[12]), float(ann[13])]
                    
                    rotation_y = float(ann[14])
                    
                    calib_list = np.reshape(calib, (12)).tolist()
                    image = cv2.imread(str(Path(image_set_path) / image_info['file_name']))
                    print(image_info['file_name'])
                    print(image.shape)
                    bbox = [float(ann[4]), float(ann[5]), float(ann[6]), float(ann[7])]
                    
                    box_3d = compute_box_3d(dim, location, rotation_y)     # box_3d形状：9*3
                    box_2d_as_point, num_keypoints, pts_center = project_to_image(box_3d, calib, image.shape)
                    box_2d_as_point = np.reshape(box_2d_as_point, (1,27))
                   
                    box_2d_as_point=box_2d_as_point.tolist()[0]
                    
                 

              
                    alpha = rotation_y - math.atan2(pts_center[0, 0] - calib[0, 2], calib[0, 0])
                    ann = {
                            'num_keypoints':num_keypoints,
                            'area':1,
                            'iscrowd': 0,
                            'keypoints': box_2d_as_point,
                            'image_id': image_id,
                            'bbox': _bbox_to_coco_bbox(bbox),
                            'category_id': cat_id,
                            'id': int(len(ret['annotations']) + 1),
                            'dim': dim,
                            'rotation_y': rotation_y,
                            'alpha': alpha,
                            'location':location,
                            'calib':calib_list}
                    ret['annotations'].append(ann)
        print("# images: ", len(ret['images']))
        print("# annotations: ", len(ret['annotations']))
        
        out_path = '{}annotations/kitti_{}.json'.format(DATA_PATH, split)
        json.dump(ret, open(out_path, 'w'))


if __name__ == '__main__':
    main()
