import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
import os
import time

# ============================================================
# 1. 配置参数
# ============================================================
MODEL_PATH = r"C:\Users\NICK\Desktop\tomato_project\best_model_MobileNetV2.pth"
DATA_DIR = r"C:\Users\NICK\Desktop\tomato_project\train"  # 使用训练集的一部分作为校准集
OUTPUT_PATH = r"C:\Users\NICK\Desktop\tomato_project\best_model_MobileNetV2_quantized.pth"

device = torch.device("cpu")  # 量化只能在 CPU 上进行
num_calibration_batches = 10  # 校准批次数，每批 32 张，共 320 张

# ============================================================
# 2. 加载原始模型
# ============================================================
print("⏳ 加载原始 FP32 模型...")
model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.classifier[1].in_features, 2)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()
print("✅ FP32 模型加载完成")

# ============================================================
# 3. 准备校准数据集（图像预处理和训练时一致）
# ============================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 加载训练集（用于校准）
calibration_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
calibration_loader = torch.utils.data.DataLoader(
    calibration_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

print(f"📊 校准数据集大小: {len(calibration_dataset)} 张")

# ============================================================
# 4. 定义量化配置
# ============================================================
# 选择后端：qnnpack 适合 ARM 架构（手机），fbgemm 适合 x86（PC）
# 因为目标是移动端部署，我们使用 qnnpack
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # 改为 fbgemm

# 准备模型进行静态量化（需要插入观察者）
model_prepared = torch.quantization.prepare(model, inplace=False)

print("🔧 开始校准...")
start_time = time.time()

# ============================================================
# 5. 校准（运行几批数据让模型统计激活值的范围）
# ============================================================
with torch.no_grad():
    for i, (images, _) in enumerate(calibration_loader):
        if i >= num_calibration_batches:
            break
        # 注意：校准数据不需要标签
        model_prepared(images)
        print(f"  校准批次 {i+1}/{num_calibration_batches} 完成")

print(f"⏱️ 校准完成，耗时 {time.time()-start_time:.2f}s")

# ============================================================
# 6. 转换为量化模型
# ============================================================
model_quantized = torch.quantization.convert(model_prepared, inplace=False)

print("✅ 量化模型转换完成")

# ============================================================
# 7. 保存量化后的模型
# ============================================================
torch.save(model_quantized.state_dict(), OUTPUT_PATH)
print(f"💾 量化模型已保存至: {OUTPUT_PATH}")

# ============================================================
# 8. （可选）对比量化前后的模型大小
# ============================================================
size_fp32 = os.path.getsize(MODEL_PATH) / (1024 * 1024)
size_int8 = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
print(f"📊 FP32 模型大小: {size_fp32:.2f} MB")
print(f"📊 INT8 量化模型大小: {size_int8:.2f} MB")
print(f"📉 压缩比: {size_fp32 / size_int8:.2f}x")

# ============================================================
# 9. 验证量化模型准确率（可选，替代速度测试）
# ============================================================
print("\n⏳ 验证量化模型准确率（使用少量测试图片）...")

# 加载测试集
test_dir = r"C:\Users\NICK\Desktop\tomato_project\test"
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# 评估 FP32 模型
fp32_acc = evaluate_model(model, test_loader)
print(f"📊 FP32 模型测试集准确率: {fp32_acc*100:.2f}%")

# 评估量化模型（注意：量化模型需要设置后端）
torch.backends.quantized.engine = 'fbgemm'  # 改为 fbgemm
int8_acc = evaluate_model(model_quantized, test_loader)
print(f"📊 INT8 量化模型测试集准确率: {int8_acc*100:.2f}%")
print(f"📉 准确率下降: {(fp32_acc - int8_acc)*100:.2f}%")

# ============================================================
# 10. 导出为 TorchScript（供 app.py 加载）
# ============================================================
print("\n⏳ 导出量化模型为 TorchScript...")
scripted_model = torch.jit.script(model_quantized)
scripted_model.save("best_model_MobileNetV2_quantized.pt")
print("✅ TorchScript 量化模型已保存为: best_model_MobileNetV2_quantized.pt")

print("\n🎉 量化完成！")