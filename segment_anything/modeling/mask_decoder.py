# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
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
        #num_multimask_outputs: int = 3,
        num_classes: int,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        #self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        #加上 1 个用于“单掩码模式”的 Token，总共会有 4 个 Mask Tokens
        self.num_mask_tokens = num_classes  #num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        #输出上采样层，
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        #每个 MLP 负责处理一个 mask_token
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        #一个 MLP，用于预测生成的掩码好不好
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据图像和提示嵌入预测掩码。
        针对 3 分类任务进行了调整：忽略 multimask_output 参数，始终返回 3 通道结果 [B, 3, H, W]。
        """
        # 1. 调用修改后的 predict_masks 获取 3 分类输出
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # 2. 移除原版的切片逻辑 (mask_slice)
        # 确保输出通道数由 self.num_mask_tokens (即 num_classes=3) 决定
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
        核心修复：即使没有提示词输入，也能防止 IndexError。
        """
        # --- 核心修复：手动拼接固定 Tokens (解决 IndexError 的关键) ---
        # 即使 sparse_prompt_embeddings 为空(size 0)，这里先定义的 4 个 token 也能保证序列长度不为 0
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        
        # 扩展到 Batch 维度: [B, 4, 256]
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        
        # 将提示词嵌入拼接到后面 [B, 4 + N, 256]
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # 运行 Transformer: 此时 hs 的长度至少为 4，取索引 [:, 0, :] 不再报错
        hs, src = self.transformer(image_embeddings, image_pe, tokens)
        
        # 分离输出 token
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # 上采样图像特征
        b, n, c = src.shape
        h = w = int(n**0.5)
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)

        # 生成掩码预测权重
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)

        # 执行点积生成最终掩码 [B, 3, H, W]
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, -1)).view(b, -1, h, w)

        # 预测 IoU 分数 [B, 3]
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # 现在出来的是[batch,3,x,y]但是3是3种分割结果
        # 要的是[batch, num_classes, x,y] one-hot encoded
        # 怎么样在改动比较小的情况下实现这点
        # 1. 要注意的是你现在deal with的是embeddings,所以你先print一下shape看看
        # 2. SAM很明显mask decoder有自己的逻辑，你先通过print和问AI的方式，了解每一行code的作用
        # 3. 结合了1和2的知识，你再问AI，怎么样改动代码实现你想要的功能

        print(f"1. image_embeddings shape: {image_embeddings.shape}")  # 预期：[batch, embed_dim, h, w]
        print(f"2. sparse_prompt_embeddings shape: {sparse_prompt_embeddings.shape}")  # 预期：[batch, N, embed_dim]
        print(f"3. dense_prompt_embeddings shape: {dense_prompt_embeddings.shape}")  # 预期：[batch, embed_dim, h, w]


        # Concatenate output tokens
        #output_tokens = torch.cat(...)

        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        # 扩展到 Batch 维度
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        print(f"4. tokens shape: {tokens.shape}")  # 预期：[batch, 1+num_mask_tokens+N, embed_dim]（1是IoU Token）

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape
        print(f"5. src shape (输入transformer前): {src.shape}")  # 预期：[batch*N_prompt, embed_dim, h, w]

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]
        print(f"6. hs shape (transformer输出): {hs.shape}")  # 预期：[batch*N_prompt, 1+num_mask_tokens+N, embed_dim]
        print(f"7. mask_tokens_out shape: {mask_tokens_out.shape}")  # 关键！预期：[batch*N_prompt, num_mask_tokens, embed_dim]

        # Upscale mask embeddings and predict masks using the mask tokens  图像特征上采样
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        print(f"8. upscaled_embedding shape: {upscaled_embedding.shape}")  # 预期：[batch*N_prompt, embed_dim//8, h_up, w_up]

        #形状变成 [Batch, Num_Tokens, Channel_Dim_Output]
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        print(f"9. hyper_in shape: {hyper_in.shape}")  # 预期：[batch*N_prompt, num_mask_tokens, embed_dim//8]

        # 原代码逻辑：生成原始masks
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        print(f"10. 原始masks shape: {masks.shape}")  # 原预期：[batch*N_prompt, 3, h_up, w_up]，改后要成[..., num_classes, ...]

        # 原代码：生成iou_pred并print
        iou_pred = self.iou_prediction_head(iou_token_out)
        print(f"11. iou_pred shape: {iou_pred.shape}")  # 

        # 新增：在这里插入one-hot转换
        
        # batch_size, num_classes, h, w = masks.shape
        # masks_softmax = F.softmax(masks, dim=1)  # 对类别维度做softmax
        # masks_argmax = torch.argmax(masks_softmax, dim=1, keepdim=True)  # 取每个位置的预测类别
        # masks_onehot = torch.zeros_like(masks_softmax).scatter_(1, masks_argmax, 1.0)  # 转one-hot格式

        # 原代码：修改return，返回one-hot后的mask
        # return masks_onehot, iou_pred  


        # # Generate mask quality predictions
        #iou_pred = self.iou_prediction_head(iou_token_out)
        # print(f"11. iou_pred shape: {iou_pred.shape}")  # 预期：[batch*N_prompt, num_mask_tokens]

        return masks, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
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
            x = F.sigmoid(x)
        return x
