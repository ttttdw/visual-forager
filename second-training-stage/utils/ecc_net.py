from utils.ecc_pooling import EccPool
import torch
from torch import nn
import numpy as np
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor


def load_eccNet(img_size):
    # 加载预训练的 VGG16 模型
    vgg_model = models.vgg16(pretrained=True)

    # 创建一个新的模型，仅包含卷积层部分，去掉最后的全连接层
    vgg_conv = nn.Sequential(*list(vgg_model.features.children()))

    # 设置为评估模式，不使用 dropout
    vgg_conv.eval()

    # 示例输入张量，需要与 VGG 模型的输入尺寸匹配（通常是 224x224）
    input_tensor = torch.randn(img_size)

    # 存储每一次池化前的特征
    pooling_features = []

    # 遍历每一层，提取池化前的特征
    def pooling_hook(module, input, output):
        pooling_features.append(input[0].shape)

    # 注册 hook，使得每一次池化前的特征都被存储
    hook_handles = []
    for layer in vgg_conv.children():
        if isinstance(layer, nn.MaxPool2d):
            handle = layer.register_forward_hook(pooling_hook)
            hook_handles.append(handle)

    # 前向传播，模型输出会存储在 pooling_features 列表中
    vgg_conv(input_tensor)

    # 注销所有注册的 hook
    for handle in hook_handles:
        handle.remove()
    # 加载预训练的 VGG 模型
    vgg_model = models.vgg16(pretrained=True)
    # 创建新的 VGG 模型，并替换所有最大池化层为自定义平均池化层

    ecc_slope = [0, 0, 3.5 * 0.02, 8 * 0.02, 16 * 0.02]
    deg2px = [round(60.0),
              round(60.0 / 2),
              round(60.0 / 4),
              round(60.0 / 8),
              round(60.0 / 16)]
    depth = 0
    for name, layer in vgg_model.features.named_modules():
        if isinstance(layer, nn.MaxPool2d):
            # 将最大池化层替换为自定义的eccentricity池化层
            if depth > 1:
                ecc_pool = EccPool(
                    input_shape=pooling_features[depth], deg2px=deg2px[depth], ecc_slope=ecc_slope[depth])
                setattr(vgg_model.features, name, ecc_pool)
                layer = ecc_pool
            depth += 1
    # ecc_net = create_feature_extractor(
    #     vgg_model, return_nodes={"features": "features.29"}
    # )
    return vgg_model.features


# ecc_net = load_eccNet(img_size=[1, 3, 256, 256])
# print(ecc_net)
# print(ecc_net(torch.rand((1, 3, 256, 256)))["features.30"].shape)
