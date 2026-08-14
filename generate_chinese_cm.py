import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

# ========== 中文字体设置 ==========
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# ========== 配置 ==========
data_dir = r"C:\Users\NICK\Desktop\tomato_project\tomato_multiclass"
model_path = "best_model_multiclass.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ========== 加载类别名称 ==========
categories = sorted([d for d in os.listdir(os.path.join(data_dir, 'train')) if os.path.isdir(os.path.join(data_dir, 'train', d))])
num_classes = len(categories)
print(f"类别数: {num_classes}")

# ========== 模型加载 ==========
model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.classifier[1].in_features, num_classes)
)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# ========== 数据预处理 ==========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ========== 加载测试集 ==========
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

print(f"测试集大小: {len(test_dataset)} 张")

# ========== 推理 ==========
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ========== 计算准确率 ==========
acc = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"测试集准确率: {acc:.4f}")

# ========== 生成混淆矩阵（中文标签） ==========
# 将英文类别名映射为中文（手动定义，可根据需要调整）
chinese_names = [
    '细菌性斑点病', '早疫病', '晚疫病', '叶霉病', '斑枯病',
    '蜘蛛螨', '靶斑病', '黄曲叶病毒', '花叶病毒', '健康'
]
# 确保顺序与 categories 一致（即按字母排序后的顺序）
# 如果顺序不对，可手动调整或直接使用 categories

# 使用 categories 作为标签（英文），但如果要显示中文，需要映射
# 此处我们直接用中文名作为标签
labels_chinese = chinese_names  # 需确认顺序与 categories 一致

cm = confusion_matrix(all_labels, all_preds)

# ========== 绘图 ==========
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels_chinese, yticklabels=labels_chinese)
plt.title('番茄病害多分类混淆矩阵', fontsize=16)
plt.xlabel('预测', fontsize=14)
plt.ylabel('真实', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix_multiclass_CH.png', dpi=300)
plt.show()

print("✅ 中文混淆矩阵已保存为 confusion_matrix_multiclass_CH.png")

# ========== 同时生成中文分类报告 ==========
report = classification_report(all_labels, all_preds, target_names=labels_chinese)
print("\n中文分类报告:\n", report)
with open('classification_report_CH.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print("✅ 中文分类报告已保存为 classification_report_CH.txt")