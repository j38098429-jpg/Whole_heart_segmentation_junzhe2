import os, sys
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

# ------- 绝对路径 & 自检 -------
ROOT = os.path.dirname(os.path.abspath(__file__))
WEIGHT = os.path.join(ROOT, "weights", "sam_vit_h_4b8939.pth")
IMAGE  = os.path.join(ROOT, "images", "truck.jpg")   # 改第二张：groceries.jpg

print("ROOT   :", ROOT)
print("WEIGHT :", WEIGHT, "->", os.path.exists(WEIGHT))
print("IMAGE  :", IMAGE,  "->", os.path.exists(IMAGE))

if not os.path.exists(WEIGHT):
    raise SystemExit("❌ 找不到权重，请确认在 weights/sam_vit_h_4b8939.pth")

if not os.path.exists(IMAGE):
    raise SystemExit("❌ 找不到图片，请确认在 images/ 下有对应文件名")

# ------- 加载模型 -------
sam = sam_model_registry["vit_h"](checkpoint=WEIGHT)
device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device)
predictor = SamPredictor(sam)

# ------- 读图并设置 -------
image = cv2.imread(IMAGE)
if image is None:
    raise SystemExit("❌ OpenCV 读图失败（可能路径/权限/大小写问题）")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)

# ------- 用点提示或框提示 -------
import numpy as np
h, w = image.shape[:2]
# 点提示
point_coords = np.array([[min(300, w-1), min(200, h-1)]], dtype=np.float32)
point_labels = np.array([1], dtype=np.int32)

masks, scores, _ = predictor.predict(
    point_coords=point_coords,
    point_labels=point_labels,
    multimask_output=True
)

print("top score:", float(scores[0]))

# ------- 可视化（无窗口环境可以保存） -------
plt.figure(figsize=(10,10))
plt.imshow(image)
plt.imshow(masks[0], alpha=0.5)
plt.axis("off")
# plt.show()  # 有桌面环境就用这句
out_path = os.path.join(ROOT, "output_mask.png")
plt.savefig(out_path, bbox_inches="tight")
print("✅ 已保存：", out_path)
