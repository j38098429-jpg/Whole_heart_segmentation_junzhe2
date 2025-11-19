import os, glob, random, math
import cv2
import numpy as np
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

# --------------------------
# 配置
# --------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(ROOT, "images")
MSK_DIR = os.path.join(ROOT, "masks")
CKPT    = os.path.join(ROOT, "weights", "sam_vit_h_4b8939.pth")
MODEL_T = "vit_h"      # 可改 vit_b / vit_l
IMG_SIZE = 1024        # SAM 默认分辨率
EPOCHS   = 5
BATCH    = 2
LR       = 1e-4
NUM_WORK = 2
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
SEED     = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# --------------------------
# 工具：采样提示（点/框）
# --------------------------
def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0: 
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    # 避免零面积
    if x0==x1: x1 = x0+1
    if y0==y1: y1 = y0+1
    return int(x0), int(y0), int(x1), int(y1)

def sample_pos_neg_points(mask: np.ndarray, n_pos=1, n_neg=1):
    h, w = mask.shape
    pos = np.argwhere(mask>0)
    neg = np.argwhere(mask==0)
    if len(pos)==0: 
        # 无前景，退化为给一个负点
        pos_points = np.empty((0,2), dtype=np.int32)
    else:
        pos_idx = np.random.choice(len(pos), size=min(n_pos, len(pos)), replace=False)
        pos_points = pos[pos_idx][:,[1,0]]  # (x,y)
    if len(neg)==0:
        neg_points = np.empty((0,2), dtype=np.int32)
    else:
        neg_idx = np.random.choice(len(neg), size=min(n_neg, len(neg)), replace=False)
        neg_points = neg[neg_idx][:,[1,0]]
    return pos_points, neg_points

# --------------------------
# 预处理（对齐 SAM 的规范）
# --------------------------
PIXEL_MEAN = np.array([123.675, 116.28, 103.53]) / 255.
PIXEL_STD  = np.array([58.395, 57.12, 57.375]) / 255.

def preprocess_image_for_sam(image_rgb: np.ndarray, img_size=IMG_SIZE):
    """仿 SamPredictor：最长边缩放到 img_size，再 0 填充到正方形；归一化到 SAM 的 mean/std。"""
    tfm = ResizeLongestSide(img_size)
    image_t = tfm.apply_image(image_rgb)
    h, w = image_t.shape[:2]
    pad_h = img_size - h
    pad_w = img_size - w
    image_padded = np.pad(image_t, ((0,pad_h),(0,pad_w),(0,0)), mode='constant', constant_values=0)

    input_image = image_padded.astype(np.float32) / 255.0
    input_image = (input_image - PIXEL_MEAN) / PIXEL_STD
    input_image = input_image.transpose(2,0,1)  # CHW
    return input_image, (h, w), tfm  # 返回缩放后尺寸和变换器

def transform_points_and_box(points_xy: np.ndarray, box_xyxy: Optional[Tuple[int,int,int,int]], tfm: ResizeLongestSide, orig_shape):
    h, w = orig_shape
    # 坐标变换到缩放后的坐标系
    if points_xy is not None and len(points_xy)>0:
        pts = points_xy.astype(np.float32)
        pts = tfm.apply_coords(pts, (h, w))
    else:
        pts = np.empty((0,2), dtype=np.float32)

    if box_xyxy is not None:
        x0,y0,x1,y1 = box_xyxy
        box = np.array([[x0,y0],[x1,y1]], dtype=np.float32)
        box = tfm.apply_coords(box, (h,w))
    else:
        box = None
    return pts, box

# --------------------------
# 数据集
# --------------------------
class SegPair(Dataset):
    def __init__(self, img_dir, msk_dir):
        self.imgs = []
        exts = (".jpg",".jpeg",".png",".bmp",".tif",".tiff")
        for p in sorted(glob.glob(os.path.join(img_dir, "*"))):
            if p.lower().endswith(exts):
                name = os.path.splitext(os.path.basename(p))[0]
                # 找匹配的 mask（优先 .png）
                for me in (".png",".jpg",".jpeg",".bmp"):
                    m = os.path.join(msk_dir, name+me)
                    if os.path.exists(m):
                        self.imgs.append((p,m))
                        break
        if not self.imgs:
            raise RuntimeError(f"在 {img_dir} / {msk_dir} 没找到成对的图像与掩码，请确认同名。")

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img_path, msk_path = self.imgs[idx]
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None: raise RuntimeError(f"读图失败：{img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        if mask is None: raise RuntimeError(f"读掩码失败：{msk_path}")
        mask_bin = (mask > 0).astype(np.uint8)

        return img_rgb, mask_bin, os.path.basename(img_path)

# --------------------------
# 损失：BCE + Dice（对 logits）
# --------------------------
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        # logits: (B,1,H,W), target: (B,1,H,W) 0/1
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        smooth = 1.0
        inter = (probs*target).sum(dim=(2,3))
        union = probs.sum(dim=(2,3)) + target.sum(dim=(2,3))
        dice = 1 - (2*inter + smooth) / (union + smooth)
        return bce + dice.mean()

# --------------------------
# 训练
# --------------------------
def train():
    print(f"Device: {DEVICE}")
    ds = SegPair(IMG_DIR, MSK_DIR)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORK, drop_last=True)

    sam: torch.nn.Module = sam_model_registry[MODEL_T](checkpoint=CKPT).to(DEVICE)
    # 冻结图像编码器（常见做法）
    for p in sam.image_encoder.parameters():
        p.requires_grad = False
    # （可选）也冻结 prompt_encoder，只训 mask_decoder
    for p in sam.prompt_encoder.parameters():
        p.requires_grad = False

    # 优化器：只训练 mask_decoder
    opt = torch.optim.AdamW(sam.mask_decoder.parameters(), lr=LR, weight_decay=0.01)
    criterion = BCEDiceLoss()
    dense_pe = sam.prompt_encoder.get_dense_pe()  # 位置编码（常量）

    sam.train()
    global_step = 0
    for epoch in range(1, EPOCHS+1):
        epoch_loss = 0.0
        for img_rgb, mask_bin, _ in dl:
            # 逐样本处理（图像尺寸各异，简化起见用循环；想提速可改成预 resize 到统一大小后再 batch）
            batch_loss = 0.0
            for i in range(img_rgb.shape[0]):
                img_np  = img_rgb[i].numpy()                     # HWC, uint8
                msk_np  = mask_bin[i].numpy().astype(np.uint8)   # HW, 0/1

                # 预处理到 SAM 输入
                inp, (h_t, w_t), tfm = preprocess_image_for_sam(img_np, IMG_SIZE)  # CHW
                im_t = torch.from_numpy(inp).to(DEVICE).unsqueeze(0)               # 1,3,1024,1024
                with torch.no_grad():
                    image_embeddings = sam.image_encoder(im_t)  # 1,256,64,64

                # 采样点 & 框，并映射到变换后坐标
                box = bbox_from_mask(msk_np)
                pos_pts, neg_pts = sample_pos_neg_points(msk_np, n_pos=1, n_neg=1)
                pts = np.vstack([pos_pts, neg_pts]) if len(neg_pts)>0 else pos_pts
                labels = np.array([1]*len(pos_pts) + [0]*len(neg_pts), dtype=np.int32)
                pts_t, box_t = transform_points_and_box(pts, box, tfm, orig_shape=msk_np.shape)

                # prompt 编码
                if len(pts_t)==0:
                    pts_t = None; labels_t=None
                else:
                    pts_t   = torch.from_numpy(pts_t).to(DEVICE)[None, :, :]           # 1,N,2
                    labels_t= torch.from_numpy(labels).to(DEVICE)[None, :]             # 1,N
                if box_t is not None:
                    box_t   = torch.from_numpy(box_t).to(DEVICE)[None, :, :]           # 1,2,2

                sparse_emb, dense_emb = sam.prompt_encoder(
                    points=(pts_t, labels_t) if pts_t is not None else None,
                    boxes=box_t if box_t is not None else None,
                    masks=None
                )

                # 解码得到低分辨率 mask（1,1,256,256）
                low_res_logits, iou_pred = sam.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=dense_pe,
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False
                )

                # 将 GT mask 缩放到 256x256（与 low_res 对齐）
                msk_resized = cv2.resize(msk_np.astype(np.float32), (256,256), interpolation=cv2.INTER_NEAREST)
                gt = torch.from_numpy(msk_resized)[None,None,:,:].to(DEVICE)  # 1,1,256,256

                loss = criterion(low_res_logits, gt)
                batch_loss += loss

            batch_loss = batch_loss / img_rgb.shape[0]
            opt.zero_grad()
            batch_loss.backward()
            opt.step()

            global_step += 1
            epoch_loss += batch_loss.item()

        avg = epoch_loss / len(dl)
        print(f"[Epoch {epoch}/{EPOCHS}] loss={avg:.4f}")

        # 每个 epoch 保存一次
        os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
        torch.save(sam.state_dict(), os.path.join(ROOT, f"runs/sam_finetuned_e{epoch}.pth"))

    print("✅ 训练完成。最新权重已保存到 runs/ 目录。")

if __name__ == "__main__":
    print("Images:", IMG_DIR)
    print("Masks :", MSK_DIR)
    if not os.path.exists(CKPT):
        raise SystemExit(f"❌ 找不到权重：{CKPT}")
    train()
