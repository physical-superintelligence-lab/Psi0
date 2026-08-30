'''
For licensing see accompanying LICENSE.txt file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
'''
import os
import cv2
import numpy as np

def map_fingers_to_colors(tf_names):
    colors = []
    for tf in tf_names:
        if 'little' in tf.lower():
            colors.append((0, 152, 191))  # light blue
        elif 'ring' in tf.lower():
            colors.append((173, 255, 47))   # green yellow
        elif 'middle' in tf.lower():
            colors.append((230, 245, 250))  # pale torquoise
        elif 'index' in tf.lower():
            colors.append((255, 99, 71))    # tomato
        elif 'thumb' in tf.lower():
            colors.append((238, 130, 238))  # violet
    return np.array(colors)

def imgs_to_mp4(img_list, mp4_path, fps=30, fourcc = cv2.VideoWriter_fourcc(*'mp4v')):
    H, W, _ = img_list[0].shape
    video_out = cv2.VideoWriter(mp4_path, fourcc,fps, (W, H))
    for img in img_list:
        BGR_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video_out.write(BGR_img) # assume input image is RGB
    video_out.release()

    os.system(f"ffmpeg -i {mp4_path} -vcodec libx264 libx264_{mp4_path} > /dev/null 2>&1")
    os.system(f"rm {mp4_path}") 

def draw_line(pointa, pointb, image, intrinsic, color=(0,255,0), thickness=5):
    # project 3d points into 2d
    pointa2, _ = cv2.projectPoints(pointa, np.eye(3), np.zeros(3), intrinsic, distCoeffs=np.zeros(5))  
    pointb2, _ = cv2.projectPoints(pointb, np.eye(3), np.zeros(3), intrinsic, distCoeffs=np.zeros(5))  
    pointa2 = pointa2.squeeze()
    pointb2 = pointb2.squeeze()

    # don't draw if the line is out of bounds
    H, W, _ = image.shape
    if (pointb2[0] < 0 and pointa2[0] > W) or (pointb2[1] < 0 and pointa2[1] > H) or (pointa2[0] < 0 and pointb2[0] > W) or (pointa2[1] < 0 and pointb2[1] > H):
        return 

    # draws a line in-place
    cv2.line(image, pointa2.astype(int), pointb2.astype(int), color=color, thickness=thickness)
    cv2.circle(image, (int(pointa2[0]), int(pointa2[1])), 15, color, -1)
    cv2.circle(image, (int(pointb2[0]), int(pointb2[1])), 15, color, -1)

def draw_line_sequence(points_list, image, intrinsic, color=(0,255,0)):
    # draw a sequence of lines in-place
    ptm = points_list[0]
    for pt in points_list[1:]:
        draw_line(ptm, pt, image, intrinsic, color)
        ptm = pt
