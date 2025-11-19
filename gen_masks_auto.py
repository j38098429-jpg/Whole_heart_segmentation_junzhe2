import os, glob, argparse
import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def to_uint8_mask(seg: np.ndarray) -> np.ndarray:
    """布尔/0-1数组 -> 0/255 uint8"""
    return (seg.astype(np.uint8) * 255)

def pick_mask(masks, strategy="largest", topk=1):
    """
    从自动生成的多个mask里挑一个/几个：
      - largest: 选面积最大的一个
      - highest_iou: 选 predicted_iou 最大的一个
      - union: 把所有 mask 取并集（如果你要得到单目标，谨慎用）
    """
    if not masks:
        return None

    if strategy == "largest":
        areas = [m["area"] for m in masks]
        seg = masks[int(np.argmax(areas))]["segmentation"]
        return to_uint8_mask(seg)

    if strategy == "highest_iou":
        ious = [m.get("predicted_iou", 0.0) for m in masks]
        seg = masks[int(np.argmax(ious))]["segmentation"]
        return to_uint8_mask(seg)

    if strategy == "union":
        H, W = masks[0]["segmentation"].shape
        union = np.zeros((H, W), dtype=np.uint8)
        for m in masks:
            union |= m["segmentation"].astype(np.uint8)
        return to_uint8_mask(union)

    # topk 取并（largest的前k个）
    if strategy.startswith("top"):
        try:
            k = int(strategy.replace("top", ""))
        except:
            k = topk
        areas = np.array([m["area"] for m in masks])
        idx = np.argsort(-areas)[:k]
        H, W = masks[0]["segmentation"].shape
        union = np.zeros((H, W), dtype=np.uint8)
        for i in idx:
            union |= masks[i]["segmentation"].astype(np.uint8)
        return to_uint8_mask(union)

    # 默认 largest
    areas = [m["area"] for m in masks]
    seg = masks[int(np.argmax(areas))]["segmentation"]
    return to_uint8_mask(seg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="images", help="图片目录")
    parser.add_argument("--masks",  default="masks",  help="输出掩码目录")
    parser.add_argument("--checkpoint", default="weights/sam_vit_h_4b8939.pth")
    parser.add_argument("--model-type", default="vit_h", choices=["vit_h","vit_l","vit_b"])
    parser.add_argument("--strategy", default="largest", choices=["largest","highest_iou","union","top3","top5"], help="挑选掩码的策略")
    parser.add_argument("--points-per-side", type=int, default=32, help="自动掩码建议的密度（越大越细）")
    parser.add_argument("--min-area", type=int, default=200, help="忽略小区域的像素数阈值")
    args = parser.parse_args()

    ensure_dir(args.masks)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=args.points_per_side,   # 覆盖密度
        pred_iou_thresh=0.86,                   # 越高越严格
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=args.min_area      # 忽略小噪点
    )

    exts = (".jpg",".jpeg",".png",".bmp",".tif",".tiff")
    imgs = [p for p in sorted(glob.glob(os.path.join(args.images, "*"))) if p.lower().endswith(exts)]
    if not imgs:
        raise SystemExit(f"在 {args.images} 没找到图片")

    for i, img_path in enumerate(imgs, 1):
        name = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(args.masks, f"{name}.png")
        print(f"[{i}/{len(imgs)}] {img_path} -> {out_path}")

        image = cv2.imread(img_path)
        if image is None:
            print(f"  跳过：读图失败")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            anns = mask_generator.generate(image)

        if not anns:
            print("  没生成到mask，可能阈值过高；可以降低 pred_iou_thresh 或 stability_score_thresh")
            continue

        mask = pick_mask(anns, strategy=args.strategy)
        cv2.imwrite(out_path, mask)
    print("✅ 全部完成。")
    
if __name__ == "__main__":
    main()
