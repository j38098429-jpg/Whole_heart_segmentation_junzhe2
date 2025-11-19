import os, glob, random, math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from segment_anything import sam_model_registry, SamPredictor

# ---------------------------
# 0. 实用函数
# ---------------------------
def set_seed(sd=42):
    random.seed(sd); np.random.seed(sd); torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)

def to_tensor_img(im: Image.Image):
    # im: PIL (RGB) -> [3, H, W], float32, 0~255(HWC 到CHW)
    arr = np.array(im.convert("RGB"), dtype=np.float32)
    arr = torch.from_numpy(arr).permute(2,0,1)  # [C,H,W]
    return arr

def to_tensor_mask(msk: Image.Image):
    # msk: PIL (L) -> [1, H, W], float32 in {0,1}
    arr = np.array(msk.convert("L"), dtype=np.uint8)
    if arr.max() > 1:  # 0/255 -> 0/1
        arr = (arr > 127).astype(np.uint8)
    arr = torch.from_numpy(arr).float()[None, ...]
    return arr

def get_box_from_mask(msk: torch.Tensor):
    # msk: [1,H,W] in {0,1}   从二值 mask 的前景像素取 外接框
    y, x = torch.where(msk[0] > 0.5)
    if y.numel() == 0:
        return None
    y0, y1 = int(y.min()), int(y.max())
    x0, x1 = int(x.min()), int(x.max())
    # +1 是为了覆盖到 max 像素
    return [x0, y0, x1+1, y1+1]

def sample_pos_neg_points(msk: torch.Tensor, n_pos=1, n_neg=0):#从 GT 中随机采样若干正点（在前景里）、负点（在背景里），作为 point-prompt 输入 SAM。
    H, W = msk.shape[-2:]
    pos = (msk[0] > 0.5).nonzero(as_tuple=False)
    neg = (msk[0] <= 0.5).nonzero(as_tuple=False)
    pts = []; lbs = []
    if pos.numel() > 0 and n_pos > 0:
        idx = torch.randint(0, pos.shape[0], (n_pos,))
        for i in idx: 
            y, x = pos[i].tolist()
            pts.append([x, y]); lbs.append(1)
    if neg.numel() > 0 and n_neg > 0:
        idx = torch.randint(0, neg.shape[0], (n_neg,))
        for i in idx:
            y, x = neg[i].tolist()
            pts.append([x, y]); lbs.append(0)
    if len(pts) == 0:
        # 兜底：中心点当正例
        pts.append([W//2, H//2]); lbs.append(1)
    return np.array(pts, dtype=np.float32), np.array(lbs, dtype=np.int32)

# ---------------------------
# 1. 数据集
# ---------------------------
class SimpleSegDataset(Dataset):
    def __init__(self, img_dir, msk_dir, size=1024):
        self.imgs = sorted(glob.glob(str(Path(img_dir)/"*")))
        self.msks = sorted(glob.glob(str(Path(msk_dir)/"*")))
        assert len(self.imgs)==len(self.msks) and len(self.imgs)>0, "images/masks 数量或文件名不匹配"
        self.size = size

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        im = Image.open(self.imgs[idx]).convert("RGB")
        mk = Image.open(self.msks[idx]).convert("L")
        # 统一缩放到 1024 x 1024（与 SAM 的预处理对齐）
        im = im.resize((self.size, self.size), Image.BILINEAR)
        mk = mk.resize((self.size, self.size), Image.NEAREST)
        img = to_tensor_img(im)      # [3,1024,1024], float32 0~255
        mask = to_tensor_mask(mk)    # [1,1024,1024], float32 0/1
        meta = {
            "path": self.imgs[idx]
        }
        return img, mask, meta

# ---------------------------
# 2. 训练（只微调 prompt_encoder + mask_decoder）
# ---------------------------
def bce_dice_loss(logits, target):
    # logits: [B,1,h,w] raw; target: [B,1,h,w] in {0,1}
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    smooth = 1.0
    inter = (probs*target).sum(dim=(1,2,3))
    union = probs.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = 1 - (2*inter + smooth)/(union + smooth)
    return bce + dice.mean()

def train_one_epoch(predictor: SamPredictor, model, loader, opt, device):
    model.train()
    total = 0.0
    for img, gt_mask, meta in loader:
        img = img.to(device)                 # [B,3,1024,1024], 0~255
        gt_mask = gt_mask.to(device)         # [B,1,1024,1024]

        # --- 只算一次图像 embedding；image_encoder 冻结即可 ---
        # SamPredictor 的 set_image 会在 no_grad 下计算 image_embedding（默认实现）
        predictor.set_image( (img[0].permute(1,2,0).cpu().numpy()).astype(np.uint8) )  # 这里用单张/BS=1 更简单稳定
        # 从 GT 生成 prompt
        box = get_box_from_mask(gt_mask[0].cpu())
        pts, lbls = sample_pos_neg_points(gt_mask[0].cpu(), n_pos=1, n_neg=1)

        # 转 tensor（predict_torch 吃的是 torch.tensor）
        pt_t = torch.tensor(pts, device=device)[None, ...]         # [1, P, 2]
        lb_t = torch.tensor(lbls, device=device)[None, ...]        # [1, P]
        box_t = torch.tensor(box, device=device)[None, ...].float() if box is not None else None

        # --- 前向：拿 decoder 的 low-res logits 来监督（梯度会回传到 prompt_encoder + mask_decoder）---
        masks, ious, low_res = predictor.predict_torch(
            point_coords=pt_t, point_labels=lb_t, boxes=box_t, multimask_output=False
        )  # masks:[1,1,H,W]；low_res:[1,1,256,256]
        # 对 low-res 监督，更稳定；也可上采样到1024监督
        loss = bce_dice_loss(low_res, F.interpolate(gt_mask, size=low_res.shape[-2:], mode="nearest"))

        opt.zero_grad()
        loss.backward()
        opt.step()

        total += loss.item()
    return total / max(1, len(loader))

def main():
    set_seed(42)
    ROOT = Path(__file__).resolve().parent
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 数据
    ds = SimpleSegDataset(
        img_dir=ROOT/"dataset/images",
        msk_dir=ROOT/"dataset/masks",
        size=1024
    )
    # 先用 BS=1，保证 predictor.set_image 的行为简单且稳定
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)

    # 2) 模型
    ckpt = ROOT/"weights/sam_vit_h_4b8939.pth"
    model = sam_model_registry["vit_h"](checkpoint=str(ckpt))
    model.to(device)

    # 冻结图像编码器（ViT）
    for p in model.image_encoder.parameters():
        p.requires_grad = False

    # 只训练 prompt_encoder + mask_decoder
    train_params = list(model.prompt_encoder.parameters()) + list(model.mask_decoder.parameters())
    opt = torch.optim.AdamW(train_params, lr=1e-4, weight_decay=1e-4)

    # 用 SamPredictor 做推理（其内部持有 model，set_image 不会对 image_encoder 反传）
    predictor = SamPredictor(model)

    epochs = 20
    for e in range(1, epochs+1):
        loss = train_one_epoch(predictor, model, loader, opt, device)
        print(f"[Epoch {e:02d}] loss = {loss:.4f}")

        # 每若干轮保存一次
        if e % 5 == 0:
            out = ROOT/f"weights/sam_finetuned_e{e}.pth"
            torch.save(model.state_dict(), out)
            print(f"✅ saved: {out}")

    # 最终保存
    final = ROOT/"weights/sam_finetuned_final.pth"
    torch.save(model.state_dict(), final)
    print(f"✅ saved final: {final}")

if __name__ == "__main__":
    main()
