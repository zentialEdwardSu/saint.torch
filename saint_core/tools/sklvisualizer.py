import cv2
import numpy as np
from typing import Optional

def skeletonpainter(image, keypoints, definition:str="coco",pattern:Optional[list]=None):
    if definition == "openpose":
        
        skeleton = [
            [1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [0, 15], [15, 17],
            [0, 16], [16, 18], [14, 19], [19, 20], [14, 21], [11, 22], [11, 23],
            [11, 24], [8, 9], [9, 10], [8, 12], [12, 13]
        ]
    elif definition == "coco":
        
        skeleton = [
            [0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
            [17, 11], [17, 12], [11, 13], [12, 14], [13, 15], [14, 16]
        ]
    else:
        if pattern:
            skeleton = pattern
        else:
            print("Invalid definition. Please choose 'openpose' or 'coco'.")
            return

    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (255, 128, 0), (128, 255, 0), (0, 128, 255), (255, 0, 128),
        (128, 0, 255), (0, 255, 128), (255, 128, 128), (128, 255, 128),
        (128, 128, 255), (255, 255, 128), (255, 128, 255), (128, 255, 255)
    ]

    # draw skeleton-line
    for i, connection in enumerate(skeleton):
        start_point = keypoints[connection[0]]
        end_point = keypoints[connection[1]]
        if start_point and end_point:
            color = colors[i % len(colors)]
            cv2.line(image, tuple(start_point), tuple(end_point), color, 2)

    # draw joint
    for i, keypoint in enumerate(keypoints):
        if keypoint:
            color = colors[i % len(colors)]
            cv2.circle(image, tuple(keypoint), 4, color, -1)
    return image

def visualizer():
    pass