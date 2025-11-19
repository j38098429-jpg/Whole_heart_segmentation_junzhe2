import torch
import torchvision
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

#加载模型和预测器
import os
from segment_anything import sam_model_registry, SamPredictor

ROOT = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(ROOT, "weights", "sam_vit_h_4b8939.pth")

sam = sam_model_registry["vit_h"](checkpoint=model_path)
predictor = SamPredictor(sam)


#读取图像
image = cv2.imread('images/truck.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)
plt.imshow(image)
plt.axis('off')
plt.show()

#用点提示分割
point_coords = np.array([[500, 375]])  # (x, y)
point_labels = np.array([1])           # 1 表示前景
masks, scores, logits = predictor.predict(
    point_coords=point_coords,
    point_labels=point_labels,
)
#显示结果
def show_mask(mask, ax, random_color=False):
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0) if random_color \
            else np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white')
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white')

fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(image)
show_mask(masks[0], ax)
show_points(point_coords, point_labels, ax)
plt.axis('off')
plt.show()

#用框提示分割
box_coords = np.array([[300, 200, 600, 400]])  # [x0, y0, x1, y1]
masks, scores, logits = predictor.predict(
    box=box_coords
)

fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(image)
show_mask(masks[0], ax)
x0, y0, x1, y1 = box_coords[0]
rect = plt.Rectangle((x0, y0), x1-x0, y1-y0, edgecolor='green', facecolor=(0,0,0,0), lw=2)
ax.add_patch(rect)
plt.axis('off')
plt.show()

#保存掩码结果
mask = masks[0].astype(np.uint8) * 255
cv2.imwrite('masks/truck.png', mask)
print("保存成功！路径：masks/truck.png")
    