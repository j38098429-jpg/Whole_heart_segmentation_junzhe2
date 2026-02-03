# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Type
from .common import LayerNorm2d

class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_classes: int = 3,  # 核心修改：适配 LV, Myocardium, Background
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        预测给定图像和提示嵌入的掩码。
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        # 定义分类所需的 Tokens
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_classes  # 3 个类别
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # 图像特征上采样层
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )

        # 为每个类别定义独立的预测头 (Hypernetworks)
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        # IoU 质量预测头
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：始终返回 3 通道的分类结果。
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测掩码的具体逻辑。
        """
        # --- 核心修复：手动拼接固定 Tokens (解决之前 main.ipynb 里的 IndexError) ---
        # 即使 sparse_prompt_embeddings 为空，也要保证 tokens 长度不为 0
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        
        # 扩展到 Batch 维度 [B, 4, 256] (1个IoU + 3个类别Token)
        batch_size = sparse_prompt_embeddings.size(0)
        output_tokens = output_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 拼接提示词嵌入 [B, 4 + N, 256]
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # 运行 Transformer
        # 此时 hs 的长度至少为 4，hs[:, 0, :] 不会再报错
        hs, src = self.transformer(image_embeddings, image_pe, tokens)
        
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # 图像特征上采样并融合密集提示
        b, c, h, w = image_embeddings.shape
        # 将 transformer 输出的 src [B, L, C] 还原为 [B, C, H, W] 
        # (假设原始 embedding 为 64x64, 上采样后为 256x256)
        src = src.transpose(1, 2).view(b, c, h, w)
        src = src + dense_prompt_embeddings
        upscaled_embedding = self.output_upscaling(src)

        # 生成 3 分类掩码权重
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)

        # 执行点积生成最终掩码结果 [B, 3, H_up, W_up]
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, -1)).view(b, -1, h, w)

        # 预测 IoU
        iou_pred = self.iou_prediction_head(iou_token_out)
        
        # 汇报用：打印关键维度
        # print(f"Decoder Output -> Masks: {masks.shape}, IoU: {iou_pred.shape}")

        return masks, iou_pred

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
