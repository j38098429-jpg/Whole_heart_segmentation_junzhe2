# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Type
from .common import LayerNorm2d

#
class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_classes: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        # 👇 这些参数其实没用了，但保留着防止报错
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        改造后的 Decoder：专注于 Multi-Class Semantic Segmentation
        不再预测 IoU，不再进行歧义性选择。
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_classes = num_classes

        self.num_mask_tokens = num_classes

        # 1. 【手术】移除 IoU Token
        # self.iou_token = nn.Embedding(1, transformer_dim) <--- 删掉它！
        
        # 2. 【重定义】这里的 mask_tokens 现在就是“类别锚点” (Class Anchors)
        # Token[0] -> 负责找背景
        # Token[1] -> 负责找左心室
        # Token[2] -> 负责找心肌
        self.class_embeddings = nn.Embedding(self.num_classes, transformer_dim)

        # 3. 图像特征上采样层 (保留原样)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )

        # 4. 每个类别独立的 MLP (保留原样)
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_classes)
            ]
        )

        # 5. 【手术】移除 IoU 预测头
        # self.iou_prediction_head = ... <--- 删掉它！

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # 直接调用预测逻辑
        masks = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )
        
        # 为了兼容 sam.py 的接口 (它期望返回两个值)，我们返回一个假的 IoU
        # 形状 [B, num_classes]
        batch_size = masks.shape[0]
        dummy_iou = torch.ones(batch_size, self.num_classes, dtype=masks.dtype, device=masks.device)
        
        return masks, dummy_iou

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        
        # 1. 【核心修改】不再拼接 IoU Token
        # output_tokens 就是我们的 3 个类别查询向量
        output_tokens = self.class_embeddings.weight
        
        # 扩展到 Batch 维度 [B, 3, 256]
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        
        # 拼接提示词 (BBox) -> [B, 3 + N, 256]
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # 2. 运行 Transformer
        # 它现在的任务是：结合 BBox 的位置信息，去图像里寻找 3 种特定的特征
        hs, src = self.transformer(image_embeddings, image_pe, tokens)
        
        # 3. 提取输出
        # hs 的前 3 个 token 就是我们要的类别特征
        # 这里的 embedding 代表了模型对 "背景"、"LV"、"Myo" 的理解
        class_tokens_out = hs[:, 0 : self.num_classes, :]

        # 4. 上采样图像特征 (Pixel Features)
        b, c, h, w = image_embeddings.shape
        src = src.transpose(1, 2).view(b, c, h, w)
        src = src + dense_prompt_embeddings
        upscaled_embedding = self.output_upscaling(src)

        # 5. 生成 Mask
        # 每个类别用自己的 MLP 生成一个权重向量，去和图像特征做点积
        hyper_in_list = []
        for i in range(self.num_classes):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](class_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)

        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, -1)).view(b, -1, h, w)

        return masks

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x
