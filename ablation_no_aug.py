# ====================================================================
# 第一部分：导入必要的库
# ====================================================================

import torch  # PyTorch 核心库，张量计算和自动求导
import torch.nn as nn  # nn 模块，提供各种神经网络层（如 Linear, Conv2d）
import torch.optim as optim  # optim 模块，提供优化器（如 Adam, SGD）
from torchvision import datasets, transforms, models  
# datasets: 提供标准数据集加载工具（如 ImageFolder）
# transforms: 数据预处理和增强工具
# models: 提供预训练模型（如 ResNet18, MobileNetV2）
from torch.utils.data import DataLoader  
# DataLoader: 批量加载数据，支持多线程和多进程
import os  # 文件和路径操作
import time  # 计时，用于统计训练时间和推理速度
import matplotlib.pyplot as plt  # 绘图库，用于生成对比图表
import matplotlib          # 导入核心模块，用于 rcParams 全局设置
import numpy as np  # 数值计算库，用于数组操作
import pandas as pd  # 数据处理库，用于生成 CSV 报告
from PIL import Image  # Python Imaging Library，用于图片读取和基本操作


# ====================================================================
# 第二部分：全局字体设置（解决图表中文乱码和字号问题）
# ====================================================================

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
# 指定 matplotlib 使用微软雅黑或黑体，以正确显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False
# 解决负号（-）显示为方块的问题
matplotlib.rcParams['font.size'] = 12
# 全局基础字号
matplotlib.rcParams['axes.labelsize'] = 14
# 坐标轴标签字号
matplotlib.rcParams['axes.titlesize'] = 16
# 子图标题字号
matplotlib.rcParams['xtick.labelsize'] = 12
# X 轴刻度字号


# ====================================================================
# 第三部分：设备配置
# ====================================================================

device = torch.device("cpu")
print(f"使用设备: {device}")
# 打印当前使用的设备，方便确认


# ====================================================================
# 第四部分：数据预处理与增强（transform）
# ====================================================================

# --- 训练集的数据预处理（包含数据增强）---
transform_train = transforms.Compose([
    # transforms.Compose 将多个变换操作组合成一个流水线
    
    transforms.Resize((224, 224)),
    # 将所有图片缩放到 224×224 像素
    # 因为预训练模型（如 ResNet18, MobileNetV2）期望输入尺寸为 224×224
    
    #transforms.RandomHorizontalFlip(p=0.5),#消融实验：关掉
    # 随机水平翻转，概率 50%
    # 数据增强：让模型看到“左右镜像”的叶片，增强泛化能力
    
   # transforms.RandomRotation(degrees=15),#消融实验：关掉
    # 随机旋转，范围 ±15 度
    # 数据增强：模拟现实中叶片不同角度的拍摄
    
    transforms.ToTensor(),
    # 将 PIL 图像（像素值 0~255）转换为 PyTorch Tensor
    # 同时将像素值从 [0, 255] 缩放到 [0.0, 1.0]
    
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
    # 标准化：将像素值从 [0,1] 转换为均值为 0、标准差为 1 的分布
    # 使用的 mean 和 std 是 ImageNet 数据集的统计值
    # 因为预训练模型就是在 ImageNet 上训练的，输入分布需要一致
    # 三个通道分别是 R, G, B
])

# --- 验证集和测试集的数据预处理（不包含数据增强）---
transform_val_test = transforms.Compose([
    transforms.Resize((224, 224)),
    # 和训练集一样缩放到 224×224
    
    transforms.ToTensor(),
    # 转换为 Tensor
    
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
    # 标准化参数和训练集保持一致
    # 注意：验证集和测试集不做随机翻转/旋转，以保证评估结果稳定可重复
])


# ====================================================================
# 第五部分：加载数据集
# ====================================================================

data_dir = r"C:\Users\NICK\Desktop\tomato_project"
# 项目根目录路径，所有子文件夹（train/val/test）都在这个目录下

# --- 创建数据集对象 ---
train_dataset = datasets.ImageFolder(
    os.path.join(data_dir, 'train'), 
    transform=transform_train
)
# ImageFolder 是 torchvision 提供的通用数据集加载器
# 它会自动扫描子文件夹，子文件夹名即为类别名（如 'healthy', 'diseased'）
# 每个子文件夹下的图片都归属于该类别
# 返回的数据格式：(image_tensor, label_index)
# label_index 根据子文件夹名的字母顺序自动编号（0, 1, 2, ...）

val_dataset = datasets.ImageFolder(
    os.path.join(data_dir, 'val'), 
    transform=transform_val_test
)
# 验证集，使用不含数据增强的 transform

test_dataset = datasets.ImageFolder(
    os.path.join(data_dir, 'test'), 
    transform=transform_val_test
)
# 测试集，同样使用不含数据增强的 transform

# --- 创建数据加载器（DataLoader）---
train_loader = DataLoader(
    train_dataset, 
    batch_size=32,  # 每批加载 32 张图片
    shuffle=True,   # 每个 epoch 打乱数据顺序，防止模型记住顺序
    num_workers=0   # 使用 0 个子进程加载数据（Windows 下建议设为 0，避免多进程报错）
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=32, 
    shuffle=False,  # 验证集不需要打乱
    num_workers=0
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=32, 
    shuffle=False,  # 测试集也不需要打乱
    num_workers=0
)

# 打印数据集大小，确认数据加载正确
print(f"训练集: {len(train_dataset)} 张")
print(f"验证集: {len(val_dataset)} 张")
print(f"测试集: {len(test_dataset)} 张")


# ====================================================================
# 第六部分：定义训练函数
# ====================================================================

def train_model(model, model_name, epochs=15):
    """
    训练单个模型并返回性能指标
    
    参数:
        model: PyTorch 模型对象
        model_name: 模型名称（用于保存文件和打印日志）
        epochs: 训练轮数，默认 15
    
    返回:
        dict: 包含训练结果各项指标的字典
    """
    print(f"\n{'='*50}")
    print(f"开始训练: {model_name}")
    print(f"{'='*50}")
    
    # --- 定义损失函数和优化器 ---
    criterion = nn.CrossEntropyLoss()
    # 交叉熵损失函数，用于多分类任务
    # 它会自动对模型输出做 softmax，然后计算交叉熵
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # Adam 优化器，一种自适应学习率的梯度下降算法
    # model.parameters() 返回模型所有可训练的参数
    # lr=0.0001 是学习率（learning rate），控制每次更新的步长
    # 注意：0.0001 是一个较小且保守的学习率，适合微调预训练模型
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )
    # 学习率调度器：当验证集准确率连续 patience 轮不提升时
    # 将学习率乘以 factor（这里乘以 0.5，即缩小一半）
    # 这有助于在接近最优值时精细化调整，防止震荡
    
    # --- 初始化记录变量 ---
    best_val_acc = 0.0
    # 保存最高的验证集准确率
    train_losses = []
    # 记录每一轮的训练损失
    val_accs = []
    # 记录每一轮的验证准确率
    epoch_times = []
    # 记录每一轮训练耗时
    
    # --- 开始训练循环 ---
    for epoch in range(epochs):
        start_time = time.time()
        # 记录本轮开始时间
        
        # ----- 训练阶段 -----
        model.train()
        # 将模型设置为训练模式
        # 这会启用 Dropout、BatchNorm 等层的训练行为
        
        running_loss = 0.0
        # 累计本轮的损失值
        
        for images, labels in train_loader:
            # 从 DataLoader 中迭代获取一个 batch 的数据
            # images: [batch_size, 3, 224, 224] 的 Tensor
            # labels: [batch_size] 的 Tensor，每个元素是类别索引 (0 或 1)
            
            images, labels = images.to(device), labels.to(device)
            # 将数据移动到指定的设备（CPU 或 GPU）
            
            optimizer.zero_grad()
            # 清空之前累积的梯度
            # 因为 PyTorch 的梯度是默认累加的，每个 batch 前必须清零
            
            outputs = model(images)
            # 前向传播：将图片输入模型，得到预测输出
            # outputs 的形状是 [batch_size, 2]，每个样本有两个类别的 logit 分数
            
            loss = criterion(outputs, labels)
            # 计算损失值：比较预测输出和真实标签
            
            loss.backward()
            # 反向传播：计算损失对模型各参数的梯度
            
            optimizer.step()
            # 更新模型参数：根据梯度进行一步优化
            
            running_loss += loss.item()
            # loss.item() 取出 loss 的标量值，累加到本轮总损失中
        
        avg_loss = running_loss / len(train_loader)
        # 计算本轮的平局损失 = 总损失 / batch 数量
        train_losses.append(avg_loss)
        # 记录到列表中
        
        # ----- 验证阶段 -----
        model.eval()
        # 将模型设置为评估模式
        # 这会禁用 Dropout，使用 BatchNorm 的全局统计量
        
        correct, total = 0, 0
        # correct: 预测正确的样本数
        # total: 总样本数
        
        with torch.no_grad():
            # 禁用梯度计算，节省内存和计算资源
            # 验证阶段不需要反向传播，所以可以安全禁用
            
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                # torch.max 返回 (最大值, 索引)
                # 这里取索引，即预测的类别（0 或 1）
                # 下标 1 表示沿着类别维度取最大值
                
                total += labels.size(0)
                # 累加本 batch 的样本数
                correct += (predicted == labels).sum().item()
                # 统计预测正确的数量
        
        val_acc = correct / total
        # 计算验证集准确率
        val_accs.append(val_acc)
        # 记录到列表中
        
        # ----- 动态调整学习率 -----
        scheduler.step(val_acc)
        # 根据验证集准确率决定是否降低学习率
        
        # ----- 保存最佳模型 -----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_model_{model_name}.pth')
            # 保存模型参数（state_dict 是参数名称到数值的映射）
            # 只保存参数，不保存模型结构，可以跨平台加载
            # 这样每轮只保存最好的那个版本，节省硬盘空间
        
        # ----- 记录本轮耗时 -----
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        
        # ----- 打印本轮日志 -----
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}, Time: {epoch_time:.2f}s")
    
    # ----- 训练结束后的评估 -----
    
    # 加载最佳模型参数（用于最终测试）
    model.load_state_dict(torch.load(f'best_model_{model_name}.pth'))
    # 这样模型参数是最佳验证集准确率对应的版本
    
    # 计算模型文件大小
    model_size = get_model_size(model)
    # 调用独立的函数，后面会定义
    
    # 测试集准确率
    test_acc = test_model(model)
    # 调用独立的测试函数
    
    # 推理速度测试
    inference_time = test_inference_speed(model)
    # 调用独立的速度测试函数
    
    # ----- 返回所有指标 -----
    return {
        'model_name': model_name,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'model_size_mb': model_size,
        'inference_time_ms': inference_time,
        'train_losses': train_losses,
        'val_accs': val_accs,
        'epoch_times': epoch_times
    }


# ====================================================================
# 第七部分：测试函数
# ====================================================================

def test_model(model):
    """
    在测试集上评估模型
    
    测试集是模型从未见过的数据，用于评估模型的真实泛化能力
    """
    model.eval()
    # 设置为评估模式
    
    correct, total = 0, 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = correct / total
    print(f"  测试集准确率: {acc:.4f} ({correct}/{total})")
    return acc


# ====================================================================
# 第八部分：计算模型文件大小
# ====================================================================

def get_model_size(model):
    """
    计算模型文件大小（MB）
    
    原理：先将模型参数保存到临时文件，然后读取文件大小
    这比直接数参数量更接近实际部署时的模型文件大小
    """
    torch.save(model.state_dict(), 'temp_model.pth')
    # 临时保存模型参数到硬盘
    
    size = os.path.getsize('temp_model.pth') / (1024 * 1024)
    # os.path.getsize 获取文件大小（字节）
    # 除以 1024*1024 转换为 MB
    
    os.remove('temp_model.pth')
    # 删除临时文件
    return size


# ====================================================================
# 第九部分：测试推理速度
# ====================================================================

def test_inference_speed(model, num_runs=100):
    """
    测试模型推理速度（毫秒/张）
    
    推理速度对于边缘设备（如手机、树莓派）部署非常重要
    num_runs 越多，测得的平均时间越稳定
    """
    model.eval()
    
    # 生成一个随机输入，模拟实际推理时的输入
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    # randn 生成服从标准正态分布的随机数
    
    # 预热：先跑 10 次，让 CUDA 内核初始化或 CPU 缓存预热
    # 这样测得的才是稳定状态下的速度
    for _ in range(10):
        _ = model(dummy_input)
    
    # 正式计时
    start_time = time.time()
    for _ in range(num_runs):
        _ = model(dummy_input)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs * 1000
    # 先计算平均每张图片的秒数，再乘以 1000 转换为毫秒
    return avg_time


# ====================================================================
# 第十部分：跨域测试（评估模型在真实场景图片上的表现）
# ====================================================================

def test_cross_domain(model, field_dir):
    """
    测试模型在野外实拍图（跨域数据）上的表现
    
    "跨域" 指的是训练数据（实验室白背景）和测试数据（真实农田）来自不同分布
    这是衡量模型泛化能力的重要指标
    """
    if not os.path.exists(field_dir):
        # 如果野外测试集文件夹不存在，直接跳过
        print("  未找到野外测试集，跳过跨域测试")
        return None
    
    model.eval()
    
    # 使用和验证集一样的预处理（无数据增强，标准化参数相同）
    field_dataset = datasets.ImageFolder(field_dir, transform=transform_val_test)
    field_loader = DataLoader(field_dataset, batch_size=8, shuffle=False, num_workers=0)
    # 使用较小的 batch_size（8），因为野外图片数量通常较少
    
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in field_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = correct / total if total > 0 else 0
    print(f"  跨域测试准确率: {acc:.4f} ({correct}/{total})")
    return acc


# ====================================================================
# 第十一部分：绘制对比图表
# ====================================================================

def plot_comparison(results):
    """
    绘制四张对比图：准确率对比、模型大小对比、推理速度对比、训练曲线对比
    
    results: list，包含每个模型的训练结果字典
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # 创建 2×2 的子图布局，总尺寸 14×10 英寸
    
    # --- 图1：准确率对比（柱状图） ---
    models = [r['model_name'] for r in results]
    test_accs = [r['test_acc'] for r in results]
    val_accs = [r['best_val_acc'] for r in results]
    # 提取数据
    
    x = np.arange(len(models))  # x 轴位置 [0, 1]
    width = 0.35  # 柱子的宽度
    
    axes[0, 0].bar(x - width/2, val_accs, width, label='验证集最佳', color='skyblue')
    axes[0, 0].bar(x + width/2, test_accs, width, label='测试集', color='lightcoral')
    # 画两组柱子，分别表示验证集最佳准确率和测试集准确率
    # x - width/2 和 x + width/2 让两组柱子并排显示
    
    axes[0, 0].set_xlabel('模型')
    axes[0, 0].set_ylabel('准确率')
    axes[0, 0].set_title('准确率对比')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    # alpha=0.3 让网格线透明度较高，不喧宾夺主
    
    # --- 图2：模型大小对比（柱状图） ---
    sizes = [r['model_size_mb'] for r in results]
    axes[0, 1].bar(models, sizes, color='lightgreen')
    axes[0, 1].set_xlabel('模型')
    axes[0, 1].set_ylabel('模型大小 (MB)')
    axes[0, 1].set_title('模型体积对比（轻量化核心指标）')
    for i, v in enumerate(sizes):
        axes[0, 1].text(i, v + 0.1, f'{v:.1f}MB', ha='center')
        # 在每个柱子上方标注具体数值
    axes[0, 1].grid(True, alpha=0.3)
    
    # --- 图3：推理速度对比（柱状图） ---
    speeds = [r['inference_time_ms'] for r in results]
    axes[1, 0].bar(models, speeds, color='gold')
    axes[1, 0].set_xlabel('模型')
    axes[1, 0].set_ylabel('推理时间 (ms/张)')
    axes[1, 0].set_title('推理速度对比（部署关键指标）')
    for i, v in enumerate(speeds):
        axes[1, 0].text(i, v + 0.1, f'{v:.1f}ms', ha='center')
    axes[1, 0].grid(True, alpha=0.3)
    
    # --- 图4：训练曲线对比（折线图） ---
    for r in results:
        axes[1, 1].plot(r['val_accs'], label=f"{r['model_name']} (最佳{max(r['val_accs']):.3f})")
        # 绘制每一轮的验证准确率变化曲线
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('验证集准确率')
    axes[1, 1].set_title('训练过程对比')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    # 自动调整子图间距，防止标签重叠
    
    plt.savefig('lightweight_comparison.png', dpi=300)
    # 保存为高清 PNG 图片，dpi=300 保证在论文中清晰
    plt.show()
    print("对比图表已保存为 lightweight_comparison.png")


# ====================================================================
# 第十二部分：生成对比报告（CSV + 关键结论）
# ====================================================================

def generate_report(results):
    """
    生成对比报告（CSV 文件）并在终端打印关键结论
    
    CSV 文件方便导入 Excel 或进一步数据分析
    关键结论部分会自动分析轻量化模型的优势
    """
    # --- 构建 DataFrame ---
    df = pd.DataFrame([{
        '模型': r['model_name'],
        '验证集最佳准确率': f"{r['best_val_acc']:.4f}",
        '测试集准确率': f"{r['test_acc']:.4f}",
        '模型大小 (MB)': f"{r['model_size_mb']:.2f}",
        '推理时间 (ms/张)': f"{r['inference_time_ms']:.2f}",
        '平均每轮训练时间 (s)': f"{np.mean(r['epoch_times']):.2f}"
    } for r in results])
    
    # --- 保存 CSV ---
    df.to_csv('model_comparison_report.csv', index=False, encoding='utf-8-sig')
    # encoding='utf-8-sig' 保证 Excel 正确打开中文
    
    print("\n对比报告已保存为 model_comparison_report.csv")
    print(df.to_string(index=False))
    
    # --- 打印关键结论 ---
    print("\n" + "="*60)
    print("关键对比结论：")
    print("="*60)
    
    # 准确率最高
    best_acc = max(results, key=lambda x: x['test_acc'])
    print(f"✅ 准确率最高: {best_acc['model_name']} ({best_acc['test_acc']:.4f})")
    
    # 体积最小
    smallest = min(results, key=lambda x: x['model_size_mb'])
    print(f"✅ 体积最小: {smallest['model_name']} ({smallest['model_size_mb']:.2f} MB)")
    
    # 速度最快
    fastest = min(results, key=lambda x: x['inference_time_ms'])
    print(f"✅ 推理最快: {fastest['model_name']} ({fastest['inference_time_ms']:.2f} ms/张)")
    
    # --- 轻量化优势分析（核心创新点） ---
    if len(results) >= 2:
        # 从 results 中分别找出 MobileNetV2 和 ResNet18 的结果
        mobile = next((r for r in results if 'mobilenet' in r['model_name'].lower()), None)
        resnet = next((r for r in results if 'resnet' in r['model_name'].lower()), None)
        
        if mobile and resnet:
            # 计算体积减小百分比
            size_reduction = (1 - mobile['model_size_mb'] / resnet['model_size_mb']) * 100
            # 计算速度提升百分比
            speed_up = (resnet['inference_time_ms'] / mobile['inference_time_ms'] - 1) * 100
            
            print(f"\n📊 MobileNetV2 相对于 ResNet18：")
            print(f"   - 模型体积减小 {size_reduction:.1f}% ({mobile['model_size_mb']:.1f}MB vs {resnet['model_size_mb']:.1f}MB)")
            print(f"   - 推理速度提升 {speed_up:.1f}% ({mobile['inference_time_ms']:.1f}ms vs {resnet['inference_time_ms']:.1f}ms)")
            
            # 判断准确率是否几乎持平（差距 < 2%）
            if mobile['test_acc'] >= resnet['test_acc'] * 0.98:
                print(f"   ✅ 准确率几乎持平 (差值仅 {abs(mobile['test_acc'] - resnet['test_acc']):.4f})")
                print(f"   🎯 结论：MobileNetV2 在几乎不损失精度的前提下，显著降低模型体积和推理时间，非常适合移动端部署！")
            else:
                print(f"   ⚠️ 准确率差距为 {abs(mobile['test_acc'] - resnet['test_acc']):.4f}，需根据实际场景权衡精度与效率")


# ====================================================================
# 第十三部分：Grad-CAM 可视化（可解释性分析）
# ====================================================================

def visualize_gradcam(model, image_path, model_name):
    """
    对单张图片生成 Grad-CAM 热力图
    
    Grad-CAM (Gradient-weighted Class Activation Mapping)：
    通过计算目标类别对最后一层卷积层的梯度，生成热力图
    红色区域表示模型决策时重点关注的区域
    这用于验证模型是否关注了正确的特征（如病斑区域）
    
    原理简述：
    1. 前向传播获取特征图和预测结果
    2. 反向传播获取目标类别对特征图的梯度
    3. 用梯度对特征图加权平均得到热力图
    4. 热力图叠加到原图展示
    """
    try:
        import cv2  # OpenCV，用于图像处理（颜色映射和缩放）
        
        model.eval()
        
        # --- 加载并预处理图片 ---
        img = Image.open(image_path).convert('RGB')
        # 加载图片并转换为 RGB（防止灰度图或 RGBA 图）
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img).unsqueeze(0).to(device)
        # unsqueeze(0) 增加 batch 维度：从 [C, H, W] 变为 [1, C, H, W]
        
        # --- 确定目标层 ---
        if 'mobilenet' in model_name.lower():
            # MobileNetV2 结构特殊，我们直接跳过（已在调用处处理）
            print(f"  {model_name} 结构特殊，跳过热力图生成")
            return
        else:  # ResNet18
            # ResNet18 的最后一层卷积层是 layer4 的最后一个 BasicBlock 的 conv2
            target_layer = model.layer4[-1].conv2
        
        # --- 注册 Hook 获取特征图 ---
        features = []
        def hook(module, input, output):
            features.append(output)
            # 将前向传播的特征图保存到 features 列表中
        handle = target_layer.register_forward_hook(hook)
        # register_forward_hook 在前向传播后自动调用 hook 函数
        
        # --- 前向传播 ---
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1)
        # 预测的类别
        
        loss = output[0, pred]
        # 取预测类别对应的 logit 值
        # 这相当于"让模型解释它为什么做出这个预测"
        
        # --- 反向传播 ---
        model.zero_grad()
        loss.backward()
        # 反向传播后，target_layer.grad 中存储了梯度
        
        # --- 获取梯度 ---
        gradients = target_layer.grad
        if gradients is None:
            print(f"  {model_name} Grad-CAM失败: 梯度不存在")
            handle.remove()
            return
        
        # --- 提取数据 ---
        gradients_np = gradients.cpu().numpy()[0]  # [C, H, W]
        features_np = features[0].cpu().numpy()[0]  # [C, H, W]
        
        # --- 计算 Grad-CAM 权重 ---
        weights = np.mean(gradients_np, axis=(1, 2))
        # 对每个通道的梯度做全局平均池化，得到每个通道的权重
        
        cam = np.sum(weights[:, np.newaxis, np.newaxis] * features_np, axis=0)
        # 用权重对特征图加权求和，得到热力图 [H, W]
        
        # --- 归一化到 [0, 255] ---
        cam = np.maximum(cam, 0)
        # ReLU：只保留正贡献
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        # 归一化到 [0, 1]，加小量防止除零
        cam = np.uint8(255 * cam)
        # 转为 0~255 的 uint8 图像
        
        cam = cv2.resize(cam, (224, 224))
        # 缩放到原图尺寸
        
        # --- 叠加热力图到原图 ---
        img_np = np.array(img.resize((224, 224)))
        # 原图转为 numpy 数组
        
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        # 将热力图转为伪彩色图（红-黄-蓝映射）
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        # OpenCV 默认 BGR，转为 RGB 以便 matplotlib 显示
        
        overlay = 0.5 * img_np + 0.5 * heatmap
        # 原图和热力图叠加，透明度 50%
        overlay = np.uint8(overlay)
        
        # --- 保存结果 ---
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 4, 1)
        plt.imshow(img_np)
        plt.title('原图')
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(cam, cmap='jet')
        plt.title('Grad-CAM热力图')
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        plt.imshow(overlay)
        plt.title('叠加图')
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        plt.imshow(heatmap)
        plt.title('热力图')
        plt.axis('off')
        
        plt.savefig(f'gradcam_{model_name}.png', dpi=300)
        plt.show()
        print(f"  Grad-CAM结果已保存为 gradcam_{model_name}.png")
        
        handle.remove()
        # 移除 hook，防止影响其他操作
        
    except Exception as e:
        print(f"  Grad-CAM生成失败: {e}")


# ====================================================================
# 第十四部分：主程序
# ====================================================================

if __name__ == "__main__":
    # __name__ == "__main__" 确保这段代码只在直接运行该脚本时执行
    # 如果该文件被 import 导入，则不会执行，避免误运行
    
    print("="*60)
    print("轻量化模型对比实验")
    print("="*60)
    
    # --- 定义要训练的模型列表 ---
    models_to_train = [
    {
        'name': 'MobileNetV2',
        'model': models.mobilenet_v2(pretrained=True),
        'modify_head': lambda m: setattr(m, 'classifier', 
                                         nn.Sequential(
                                             nn.Dropout(0.2),
                                             nn.Linear(m.classifier[1].in_features, 2)
                                         ))
    }
]
    
    results = []
    # 用于存储所有模型的训练结果
    
    # --- 依次训练每个模型 ---
    for config in models_to_train:
        model = config['model']
        config['modify_head'](model)
        # 替换分类头
        
        model = model.to(device)
        # 将模型移动到指定设备（CPU 或 GPU）
        
        # ----- 训练 -----
        result = train_model(model, config['name'])
        results.append(result)
        
        # ----- 跨域测试（如果有野外图片） -----
        field_dir = os.path.join(data_dir, 'field_test')
        cross_acc = test_cross_domain(model, field_dir)
        if cross_acc is not None:
            result['cross_acc'] = cross_acc
            # 将跨域测试结果存入 result 字典（虽然这里没有用到，但可以扩展）
        
        # ----- Grad-CAM 可视化（已迁移到独立脚本 fix_gradcam.py） -----
        # sample_path = os.path.join(data_dir, 'test', 'diseased', 
        #                            os.listdir(os.path.join(data_dir, 'test', 'diseased'))[0])
        # print(f"\n生成 {config['name']} Grad-CAM可视化...")
        # visualize_gradcam(model, sample_path, config['name'])
        print(f"\n⏭️ 跳过 {config['name']} Grad-CAM（请单独运行 fix_gradcam.py 生成）")
        
        # ----- 清理资源 -----
        del model
        # 删除模型对象，释放内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # 如果使用 GPU，清空缓存防止显存溢出
    
    # ----- 绘制对比图 -----
    plot_comparison(results)
    
    # ----- 生成对比报告 -----
    generate_report(results)
    
    print("\n" + "="*60)
    print("实验完成！轻量化对比结果已全部保存。")
    print("="*60)