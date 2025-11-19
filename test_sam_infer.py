import os, glob, argparse
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

def pick_image(image_arg, root):
    if image_arg:
        img_path = image_arg if os.path.isabs(image_arg) else os.path.join(root, image_arg)
        if not os.path.exists(img_path):
            raise SystemExit(f"❌ 指定图片不存在：{img_path}")
        return img_path
    # 自动选择 images/ 下第一张图片
    image_dir = os.path.join(root, "images")
    imgs = sorted(glob.glob(os.path.join(image_dir, "*.*")))
    if not imgs:
        raise SystemExit(f"❌ 没有找到图片，请放一张到 {image_dir}")
    return imgs[0]

def main():
    ROOT = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="图片路径（相对或绝对）。不填则自动选 images/ 下第一张")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(ROOT, "weights", "sam_vit_h_4b8939.pth"))
    parser.add_argument("--model-type", type=str, default="vit_h", choices=["vit_h","vit_l","vit_b"])
    parser.add_argument("--point", nargs=2, type=int, help="点提示：x y")
    parser.add_argument("--box", nargs=4, type=int, help="框提示：x0 y0 x1 y1")
    parser.add_argument("--save", action="store_true", help="无法弹窗时保存为 output_mask.png")
    args = parser.parse_args()

    image_path = pick_image(args.image, ROOT)
    print(f"✅ 使用图片：{image_path}")

    # 加载 SAM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device)
    predictor = SamPredictor(sam)

    # 读图并设置
    image = cv2.imread(image_path)
    if image is None:
        raise SystemExit(f"❌ 无法读取图片：{image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    # 构造提示（点 或 框 二选一；若都不填，默认点在图像中心）
    h, w = image.shape[:2]
    masks = None

    if args.point:
        x, y = args.point
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        point_coords = np.array([[x, y]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)  # 1=前景，0=背景
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )
        vis_type = f"POINT ({x},{y})"
    elif args.box:
        x0, y0, x1, y1 = args.box
        x0, x1 = np.clip([x0, x1], 0, w - 1)
        y0, y1 = np.clip([y0, y1], 0, h - 1)
        box = np.array([[min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]], dtype=np.float32)
        masks, scores, _ = predictor.predict(
            box=box,
            multimask_output=True
        )
        vis_type = f"BOX ({x0},{y0},{x1},{y1})"
    else:
        # 默认给图像中心一个前景点
        cx, cy = w // 2, h // 2
        point_coords = np.array([[cx, cy]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )
        print("masks shape:", masks.shape)  # 打印掩码的维度
        print("scores shape:", scores.shape)  # 打印分数的维度

        
        
        vis_type = f"POINT ({cx},{cy}) [auto]"

    # 可视化
    print("提示类型:", vis_type, "| masks shape:", masks.shape, "| top score:", float(scores[0]))
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.imshow(masks[0], alpha=0.5)
    plt.axis("off")
    
    # 打印提示类型和掩码维度
    print(f"提示类型：{vis_type}, masks shape:", masks.shape, "| top score:", float(scores[0]))

# 可视化掩码
    print("显示掩码:", vis_type, "masks shape:", masks.shape, "| top score:", float(scores[0]))

# 显示图像和掩码
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.imshow(masks[0], alpha=0.5)  # 显示第一个掩码，透明度为0.5
    plt.axis('off')
    

    if args.save:
        out_path = os.path.join(ROOT, "output_mask.png")
        plt.savefig(out_path, bbox_inches="tight")
        print(f"💾 已保存结果：{out_path}")
    else:
        try:
            plt.show()
        except Exception as e:
            out_path = os.path.join(ROOT, "output_mask.png")
            plt.savefig(out_path, bbox_inches="tight")
            print(f"⚠️ 无法显示窗口，已改为保存：{out_path}\n原因：{e}")

if __name__ == "__main__":
    main()



