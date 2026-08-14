# ============================================================
# 导入必要的库
# ============================================================
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

import torch
# PyTorch 核心库，提供张量计算、自动求导、神经网络基础功能

import torch.nn as nn
# nn 模块，包含各种神经网络层的定义（如 Linear、Conv2d、BatchNorm）

from torchvision import models, transforms
# models: 提供预训练模型（如 ResNet18、MobileNetV2）
# transforms: 数据预处理工具（缩放、裁剪、标准化等）

from PIL import Image
# Python 图像处理库，用于读取和保存图片

import numpy as np
# 数值计算库，用于数组操作（热力图本质就是二维数组）

import matplotlib.pyplot as plt
# 绘图库，用于显示和保存图片

import cv2
# OpenCV 计算机视觉库，主要用于图像的颜色映射（colormap）和缩放

import os
# 操作系统接口，用于文件和路径操作


# ============================================================
# 1. 配置参数（这是唯一需要你手动修改的地方）
# ============================================================

# data_dir：你的项目文件夹路径
# 所有训练好的模型文件（.pth）和测试图片都放在这里
data_dir = r"C:\Users\NICK\Desktop\tomato_project"

# device：计算设备
# 这里强制使用 CPU，因为你的显卡（RTX 5050）和当前 PyTorch 版本不兼容
# 如果用 CPU 跑热力图，速度完全够（只处理一张图片）
device = torch.device("cpu")


# ============================================================
# 2. 加载训练好的 ResNet18 模型
# ============================================================

# 2.1 创建模型结构
# models.resnet18(pretrained=False) 创建一个 ResNet18 模型
# pretrained=False 表示不加载 ImageNet 预训练权重
# 因为我们要加载的是你之前训练好的番茄分类权重（保存在 best_model_ResNet18.pth）
model = models.resnet18(pretrained=False)

# 2.2 修改模型的最后一层（全连接层）
# 原始 ResNet18 的全连接层是 model.fc，输入特征数 512，输出 1000（ImageNet 类别数）
# 我们的任务是二分类（健康 vs 病害），所以把输出改为 2
# model.fc.in_features 是 512，表示全连接层的输入特征数
model.fc = nn.Linear(model.fc.in_features, 2)

# 2.3 加载训练好的权重
# torch.load 读取 .pth 文件，返回一个字典（state_dict）
# state_dict 包含模型所有参数的名称和数值
# map_location=device 确保权重被加载到 CPU 上（即使原模型在 GPU 上训练）
model.load_state_dict(torch.load(
    os.path.join(data_dir, 'best_model_ResNet18.pth'),
    map_location=device
))

# 2.4 将模型移到指定设备（CPU）
model = model.to(device)

# 2.5 设置为评估模式
# model.eval() 会禁用 Dropout 和 BatchNorm 的训练行为
# 在评估/推理时，Dropout 不随机丢弃神经元，BatchNorm 使用全局统计量而非 batch 统计量
model.eval()

# 打印确认信息
print("✅ 模型加载成功！")


# ============================================================
# 3. Grad-CAM 核心函数（最关键的代码）
# ============================================================

def grad_cam(model, img_tensor, target_layer_name='layer4', target_class=None):
    """
    对单张图片生成 Grad-CAM 热力图

    原理（简单理解）：
    1. 把图片输入模型，得到预测结果
    2. 记录模型中间某一层（目标层）输出的"特征图"
    3. 对预测结果进行反向传播，计算目标层每个通道的"梯度"
    4. 用梯度作为权重，对特征图进行加权求和，得到热力图
    5. 热力图的红色区域表示"模型决策时最关注的区域"

    参数：
        model: PyTorch 模型（已加载权重，处于 eval 模式）
        img_tensor: 预处理后的图片张量，形状为 [1, 3, 224, 224]
        target_layer_name: 目标层的名称（字符串），默认 'layer4'
        target_class: 目标类别索引（0=健康，1=病害）
                      如果为 None，则自动使用模型预测的类别

    返回：
        cam: 热力图，形状为 [H, W]，数值范围 0~1
    """

    # ---------- 3.1 找到目标层 ----------
    # model.named_modules() 遍历模型的所有子模块，返回 (名称, 模块) 对
    # 例如：('layer4', Sequential对象), ('layer4.0.conv1', Conv2d对象), ...
    target_layer = None
    for name, module in model.named_modules():
        if name == target_layer_name:
            target_layer = module
            break

    # 如果找不到目标层，抛出错误并退出
    if target_layer is None:
        raise RuntimeError(f"找不到目标层: {target_layer_name}")

    # 打印目标层信息，方便调试
    print(f"📍 目标层: {target_layer_name} ({type(target_layer).__name__})")

    # ---------- 3.2 准备钩子（Hook）用于捕获中间数据 ----------
    # 钩子（Hook）是 PyTorch 的一种机制，可以在前向或反向传播时插入自定义操作
    # 这里我们需要捕获两类数据：
    #   1. 前向传播时目标层的输出（特征图）
    #   2. 反向传播时目标层的梯度

    features = []   # 用于存储特征图
    gradients = []  # 用于存储梯度

    # 前向钩子：在前向传播经过目标层时被调用
    # 参数：
    #   module: 当前层对象
    #   input: 输入张量（可忽略）
    #   output: 输出张量（即特征图）
    def forward_hook(module, input, output):
        features.append(output)  # 保存特征图

    # 反向钩子：在反向传播经过目标层时被调用
    # 参数：
    #   module: 当前层对象
    #   grad_input: loss 对输入的梯度（可忽略）
    #   grad_output: loss 对输出的梯度（即我们需要的梯度）
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])  # 保存梯度

    # 注册钩子到目标层
    # register_forward_hook: 注册前向钩子
    # register_full_backward_hook: 注册反向钩子
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    # ---------- 3.3 前向传播 ----------
    # 将图片输入模型，得到预测输出
    # output 的形状是 [1, 2]，表示 batch=1，每个类别的得分（logit）
    output = model(img_tensor)

    # ---------- 3.4 确定目标类别 ----------
    # 如果用户没有指定目标类别，就使用模型预测的类别
    # torch.argmax 返回最大值的索引，即预测的类别
    if target_class is None:
        target_class = torch.argmax(output, dim=1).item()

    # ---------- 3.5 反向传播 ----------
    # 清零之前累积的梯度，防止梯度叠加
    model.zero_grad()

    # 对目标类别的得分进行反向传播
    # output[0, target_class] 表示第 0 张图片、目标类别的得分
    # .backward() 会计算该得分对模型所有参数的梯度
    # 梯度会沿着计算图反向传播，经过目标层时，反向钩子会被触发
    output[0, target_class].backward()

    # ---------- 3.6 移除钩子 ----------
    # 钩子用完后要及时移除，避免影响后续操作或造成内存泄漏
    forward_handle.remove()
    backward_handle.remove()

    # ---------- 3.7 检查是否成功捕获数据 ----------
    if not features:
        raise RuntimeError("未捕获到特征图")
    if not gradients:
        raise RuntimeError("未捕获到梯度")

    # ---------- 3.8 提取特征图和梯度 ----------
    # features[0] 是前向钩子捕获的特征图，形状 [1, C, H, W]
    # C = 通道数（对于 layer4，C = 512），H = 特征图高度，W = 特征图宽度
    feature_map = features[0]

    # gradients[0] 是反向钩子捕获的梯度，形状 [1, C, H, W]
    # 梯度和特征图的形状完全一致，因为梯度是 loss 对特征图的导数
    grad = gradients[0]

    # ---------- 3.9 计算 Grad-CAM 热力图 ----------
    # 步骤1：计算每个通道的权重 = 梯度在空间维度（H, W）上的全局平均
    # dim=(2,3) 表示在第 2 和第 3 维度（即 H 和 W）上求平均
    # keepdim=True 保持维度不变，方便后续广播计算
    # 结果形状： [1, C, 1, 1]
    weights = torch.mean(grad, dim=(2, 3), keepdim=True)

    # 步骤2：用权重对特征图进行加权求和
    # weights * feature_map 是逐元素乘法，形状 [1, C, H, W]
    # torch.sum(..., dim=1) 在通道维度上求和，结果形状 [1, 1, H, W]
    cam = torch.sum(weights * feature_map, dim=1, keepdim=True)

    # 步骤3：应用 ReLU 激活函数
    # 只保留正贡献，负值表示对目标类别有抑制作用
    cam = torch.relu(cam)

    # 步骤4：移除 batch 和 channel 维度，转为 numpy 数组
    # 从 [1, 1, H, W] 变为 [H, W]
    cam = cam.squeeze().detach().cpu().numpy()

    # 步骤5：归一化到 [0, 1]
    # 公式：(x - min) / (max - min + 1e-8)
    # 加 1e-8 是为了防止除零（当所有值相等时）
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)

    return cam


# ============================================================
# 4. 选择一张测试图片
# ============================================================

# 4.1 构建测试图片的路径
# 从 test/diseased 文件夹中取第一张图作为示例
# os.path.join 自动拼接路径，确保跨平台兼容（Windows 用 \，Linux 用 /）
test_dir = os.path.join(data_dir, 'test', 'diseased')

# os.listdir 列出文件夹中所有文件，返回一个列表
# [0] 取第一个文件名
sample_img = os.path.join(test_dir, os.listdir(test_dir)[0])

print(f"📷 使用图片: {sample_img}")

# 4.2 加载图片
# PIL.Image.open 读取图片
# .convert('RGB') 确保图片为三通道彩色图（防止灰度图或 RGBA 图）
img_pil = Image.open(sample_img).convert('RGB')

# 4.3 数据预处理（和训练时保持一致）
# transforms.Compose 将多个预处理步骤组合成一个流水线
transform = transforms.Compose([
    # 缩放到 224x224 像素
    # ResNet18 的输入尺寸要求是 224x224
    transforms.Resize((224, 224)),

    # 转换为 PyTorch Tensor
    # 同时将像素值从 [0, 255] 缩放到 [0.0, 1.0]
    transforms.ToTensor(),

    # 标准化：将像素值从 [0, 1] 转为均值为 0、标准差为 1 的分布
    # 使用的均值和标准差是 ImageNet 数据集的统计值
    # 因为预训练模型是在 ImageNet 上训练的，输入分布需要一致
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 对图片进行预处理
# .unsqueeze(0) 增加一个 batch 维度，从 [3, 224, 224] 变为 [1, 3, 224, 224]
# .to(device) 将张量移到指定设备（CPU）
img_tensor = transform(img_pil).unsqueeze(0).to(device)


# ============================================================
# 5. 调用 grad_cam 函数生成热力图
# ============================================================

try:
    # 目标层指定为 'layer4'（ResNet18 的最后一个残差块）
    # 这是语义信息最丰富的层，适合用于 Grad-CAM
    cam = grad_cam(model, img_tensor, target_layer_name='layer4')
    print("✅ Grad-CAM 生成成功！")
except Exception as e:
    print(f"❌ Grad-CAM 生成失败: {e}")
    exit()  # 如果失败，退出程序，避免后面的代码报错


# ============================================================
# 6. 可视化热力图并保存
# ============================================================

# 6.1 准备原图
# 将 PIL 图片缩放到 224x224，转为 numpy 数组
# 方便后续和热力图叠加显示
img_np = np.array(img_pil.resize((224, 224)))

# 6.2 处理热力图
# cam 是 [H, W] 的 float 数组，值范围 0~1
# 先乘以 255，转为 uint8（0~255）
# 再用 cv2.resize 缩放到 224x224（保持和原图一致）
cam_resized = cv2.resize(np.uint8(255 * cam), (224, 224))

# 6.3 将热力图转为伪彩色图
# cv2.applyColorMap 将灰度图映射为彩色图
# cv2.COLORMAP_JET 是 JET 色图（蓝 -> 青 -> 黄 -> 红）
# 输出的颜色通道顺序是 BGR（OpenCV 默认）
heatmap_bgr = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)

# 将 BGR 转为 RGB，以便 matplotlib 正确显示
heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

# 6.4 生成叠加图
# 原图和热力图按 1:1 叠加
overlay = 0.5 * img_np + 0.5 * heatmap_rgb
overlay = np.uint8(overlay)  # 转为 uint8 以便显示

# 6.5 创建 1x4 的子图布局
# figsize=(16, 4) 表示图片总尺寸为 16 英寸宽、4 英寸高
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# 子图1：原图
axes[0].imshow(img_np)
axes[0].set_title("原图")
axes[0].axis('off')  # 不显示坐标轴

# 子图2：Grad-CAM 热力图（灰度，用 JET 色图显示）
axes[1].imshow(cam_resized, cmap='jet')
axes[1].set_title("Grad-CAM 热力图")
axes[1].axis('off')

# 子图3：伪彩色热力图
axes[2].imshow(heatmap_rgb)
axes[2].set_title("热力图 (伪彩)")
axes[2].axis('off')

# 子图4：叠加图（原图 + 热力图）
axes[3].imshow(overlay)
axes[3].set_title("叠加图")
axes[3].axis('off')

# 自动调整子图间距，防止标签重叠
plt.tight_layout()

# 保存为高清 PNG 图片
# dpi=300 保证在论文中显示清晰
# bbox_inches='tight' 自动裁剪空白边距
output_path = os.path.join(data_dir, 'ResNet18_GradCAM_Fixed.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')

# 显示图片窗口（运行时会弹出）
plt.show(block=False)  # 非阻塞模式，显示图片后立即返回终端
input("按 Enter 键关闭图片...")  # 等待用户按回车

print(f"🎉 热力图已保存为: {output_path}")