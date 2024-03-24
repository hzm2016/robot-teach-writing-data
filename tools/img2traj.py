import numpy as np
import cv2
import os
import glob
import numpy as np
import scipy.ndimage.morphology as m
import scipy.special
import time, math, copy
import bezier
import matplotlib.pyplot as plt
from .skeltonize import *
import random
import shutil
import tqdm 


def skeletonize(img, chunk_size=3, show_image=False):

    im = (img>128).astype(np.uint8)
    im = thinning(im)
    
    rects = []
    polys = traceSkeleton(im,0,0,im.shape[1],im.shape[0],chunk_size,999,rects) ### 
    
    img_canvas = np.full((128,128),255, np.uint8)

    for l in polys:
        c = (0,0,0)
        for i in range(0,len(l)-1):
            cv2.line(img_canvas,(l[i][0],l[i][1]),(l[i+1][0],l[i+1][1]),c,2)

    if show_image:
        cv2.imshow('',img_canvas);cv2.waitKey(0)

    # save the image
    cv2.imwrite('test.png', img_canvas)

    return polys, img_canvas

def _extract_points(img):
    
    return skeletonize(~img)

if __name__ == "__main__":

    file_path = '/home/cunjun/Robot-Teaching-Assiantant/gan/data/seq/train/imgs_part'
    width, height = 128, 128
    offset = 3.0
    show_animation = True
    image = np.ones((height, width)) * 255
    
    if file_path == 'line':
        x1, y1 = 20, 20
        x2, y2 = 200, 200
        line_thickness = 1
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), thickness=line_thickness)
    elif file_path == 'circle':
        center_coordinates = (100, 100)
        radius = 20
        color = [0,0,0]
        thickness = 1
        cv2.circle(image, center_coordinates, radius, color, thickness)
    elif os.path.isdir(file_path):
        pls_lst = []     
        max_length = 0   
        os.makedirs(file_path.replace('imgs_part', 'imgs_part_points'),exist_ok=True)
        for folder_name in tqdm.tqdm(sorted(glob.glob(file_path + '/*'))):
            os.makedirs(folder_name.replace('imgs_part', 'imgs_part_points'),exist_ok=True)
            for file_name in sorted(glob.glob(folder_name+'/*.jpg')):
                prefix = file_name[:-4] 
                log_name = prefix + '.txt'
                # out_file = open(log_name.replace('imgs_part', 'imgs_part_points'),'w')
                image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
                points, images = _extract_points(image)
                if len(points) == 0:
                    points = [[[0,0]]]
                points = np.array(points[0])
                if len(points) > max_length:
                    max_length = len(points) 
                    print(max_length)
                # np.savetxt(out_file, points)
                # inverse = ~images
                # if inverse.sum() == 0:
                #     print(file_name + ' is empty')
                #     shutil.copy(file_name, file_name.replace('imgs_part', 'imgs_part_points'))
                # else:
                #     cv2.imwrite(file_name.replace('imgs_part', 'imgs_part_points'), images)
    else:
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        points, image = _extract_points(image)
        file_name = file_path
        file_name = file_name.replace('png', 'jpg')
        cv2.imwrite(file_name, image)
        



