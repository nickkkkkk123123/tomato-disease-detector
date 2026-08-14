import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ========== 配置 ==========
data_dir = r"C:\Users\NICK\Desktop\tomato_project\tomato_multiclass"   # 数据划分后的目录
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 获取类别名称（按字母顺序，与 ImageFolder 自动分配一致）
categories = sorted([d for d in os.listdir(os.path.join(data_dir, 'train')) if os.path.isdir(os.path.join(data_dir, 'train', d))])
num_classes = len(categories)
print(f"类别数: {num_classes}")
print("类别:", categories)

# ========== 数据预处理 ==========
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
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

# ========== 加载数据 ==========
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform_val_test)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform_val_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

print(f"训练集: {len(train_dataset)} 张")
print(f"验证集: {len(val_dataset)} 张")
print(f"测试集: {len(test_dataset)} 张")

# ========== 模型定义 ==========
model = models.mobilenet_v2(pretrained=True)
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.classifier[1].in_features, num_classes)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

# ========== 训练 ==========
epochs = 20
best_val_acc = 0.0
train_losses = []
val_accs = []

print("\n开始训练...")
for epoch in range(epochs):
    # 训练
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

    # 验证
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
    scheduler.step(val_acc)

    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model_multiclass.pth')

    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")

# 加载最佳模型
model.load_state_dict(torch.load('best_model_multiclass.pth'))

# ... 前面的代码不变 ...

# ========== 测试集评估 ==========
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 准确率
test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"\n测试集准确率: {test_acc:.4f}")

# ========== 英文混淆矩阵 ==========
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=categories, yticklabels=categories)
plt.title('Confusion Matrix - Tomato Disease Multi-class')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix_multiclass.png', dpi=300)
plt.show()

report = classification_report(all_labels, all_preds, target_names=categories)
print("\n英文分类报告:\n", report)
with open('classification_report.txt', 'w') as f:
    f.write(report)

# ========== 中文混淆矩阵 ==========
# 确保 chinese_names 已定义，且顺序与 categories 一致
chinese_names = ['细菌性斑点病', '早疫病', '晚疫病', '叶霉病', '斑枯病',
                 '蜘蛛螨', '靶斑病', '黄曲叶病毒', '花叶病毒', '健康']
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=chinese_names, yticklabels=chinese_names)
plt.title('番茄病害多分类混淆矩阵', fontsize=16)
plt.xlabel('预测', fontsize=14)
plt.ylabel('真实', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix_multiclass_CH.png', dpi=300)
plt.show()
print("✅ 中文混淆矩阵已保存为 confusion_matrix_multiclass_CH.png")

report_ch = classification_report(all_labels, all_preds, target_names=chinese_names)
print("\n中文分类报告:\n", report_ch)
with open('classification_report_CH.txt', 'w', encoding='utf-8') as f:
    f.write(report_ch)
print("✅ 中文分类报告已保存为 classification_report_CH.txt")