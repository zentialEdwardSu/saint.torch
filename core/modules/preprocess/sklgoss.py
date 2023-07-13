import numpy as np
import cv2

def skeleton_to_heatmap(skeleton, image_size, sigma):
  """
  将骨架转换为热图。

  Args:
      skeleton: skeleton of data
      image_size: image size of transformation。
      sigma: standard deviation。

  Returns:
      the joint heatmap
  """

  # Coordinate transformation。
  skeleton_image_coordinates = skeleton / image_size

  # do goss
  heatmaps = []
  for joint in skeleton_image_coordinates:
    heatmap = cv2.GaussianBlur(
        np.array(joint, np.float32), (sigma, sigma), 0)
    heatmaps.append(heatmap)

  # stack heatmap if each joint。
  heatmap = np.sum(heatmaps, axis=0)

  return heatmap