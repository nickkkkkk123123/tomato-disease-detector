import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import time
import matplotlib.pyplot as plt
import numpy as np

# ============ 1. 设备配置 ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ============ 2. 数据预处理 ============
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),      # 数据增强：随机翻转
    transforms.RandomRotation(degrees=15),       # 数据增强：随机旋转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

transform_val_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# ============ 3. 加载数据 ============
data_dir = r"C:\Users\NICK\Desktop\tomato_project"  # 修改这里！

train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform_val_test)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform_val_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

print(f"训练集: {len(train_dataset)} 张")
print(f"验证集: {len(val_dataset)} 张")
print(f"测试集: {len(test_dataset)} 张")

# ============ 4. 加载预训练模型（MobileNetV2） ============
model = models.mobilenet_v2(pretrained=True)
# 替换最后的分类层（二分类）
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(device)

# ============ 5. 定义损失函数和优化器 ============
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 学习率调度器（当验证集准确率不再提升时，降低学习率）
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

# ============ 6. 训练循环 ============
def train():
    epochs = 20
    best_val_acc = 0.0
    train_losses, val_accs = [], []
    
    for epoch in range(epochs):
        # --- 训练阶段 ---
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # --- 验证阶段 ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = correct / total
        val_accs.append(val_acc)
        
        # 动态调整学习率
        scheduler.step(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accs)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.savefig('training_curves.png')
    plt.show()
    print("训练曲线已保存为 training_curves.png")

# ============ 7. 测试函数 ============
def test():
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = correct / total
    print(f"测试集准确率: {test_acc:.4f} ({correct}/{total})")
    return all_preds, all_labels

# ============ 8. Grad-CAM 可视化 ============
def grad_cam_visualization(image_path, model, target_layer):
    """对单张图片生成Grad-CAM热力图"""
    from torchvision import transforms
    from PIL import Image
    import cv2
    
    model.eval()
    
    # 加载并预处理图片
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # 获取目标层的梯度
    def save_gradient(grad):
        global gradients
        gradients = grad
    
    handle = target_layer.register_backward_hook(save_gradient)
    
    # 前向传播
    output = model(img_tensor)
    pred = torch.argmax(output, dim=1)
    loss = output[0, pred]
    
    # 反向传播
    model.zero_grad()
    loss.backward()
    
    # 获取特征图和梯度
    features = target_layer.output  # 需要hook保存
    gradients_np = gradients.cpu().numpy()[0]
    features_np = features.cpu().numpy()[0]
    
    # 计算权重（全局平均池化）
    weights = np.mean(gradients_np, axis=(1, 2))
    cam = np.sum(weights[:, np.newaxis, np.newaxis] * features_np, axis=0)
    
    # 归一化
    cam = np.maximum(cam, 0)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
    cam = np.uint8(255 * cam)
    cam = cv2.resize(cam, (224, 224))
    
    # 叠加到原图
    img_np = np.array(img.resize((224, 224)))
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = 0.5 * img_np + 0.5 * heatmap
    overlay = np.uint8(overlay)
    
    # 显示
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title('Original')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(cam, cmap='jet')
    plt.title('Grad-CAM')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title('Overlay')
    plt.axis('off')
    plt.savefig('gradcam_result.png')
    plt.show()
    print("Grad-CAM结果已保存为 gradcam_result.png")

# ============ 9. 跨域测试（野外实拍图） ============
def test_cross_domain(field_dir):
    """测试模型在野外实拍图上的表现"""
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    # 加载野外图片
    field_dataset = datasets.ImageFolder(field_dir, transform=transform_val_test)
    field_loader = DataLoader(field_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in field_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = correct / total if total > 0 else 0
    print(f"跨域测试（野外实拍）准确率: {acc:.4f} ({correct}/{total})")
    return acc

# ============ 10. 主程序 ============
if __name__ == "__main__":
    # 训练
    train()
    
    # 测试
    test()
    
    # 跨域测试（如果 field_test 文件夹存在且有图片）
    field_dir = os.path.join(data_dir, 'field_test')
    if os.path.exists(field_dir):
        test_cross_domain(field_dir)
    
    # Grad-CAM 可视化示例
    # 从测试集里找一张图演示
    sample_path = os.path.join(data_dir, 'test', 'diseased', os.listdir(os.path.join(data_dir, 'test', 'diseased'))[0])
    model.load_state_dict(torch.load('best_model.pth'))
    # 获取MobileNetV2的最后一层卷积层
    target_layer = model.features[-1]
    grad_cam_visualization(sample_path, model, target_layer)