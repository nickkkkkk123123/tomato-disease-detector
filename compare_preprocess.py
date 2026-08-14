import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# 1. 加载模型（原始模型）
model = torch.jit.load('model_multiclass.pt', map_location='cpu')
model.eval()

# 2. 加载图片（使用和 Web 版同一张）
image_path = r'C:\Users\NICK\Desktop\54fbb2fb43166d22b8d888d8963bc0e79152d2fd.jpeg'

# --- 方式 A：Web 版（PIL + torchvision）---
def predict_web():
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
    return probs

# --- 方式 B：模拟 Android 预处理（BILINEAR 缩放）---
def predict_android_sim():
    # 用 OpenCV 读取图片（BGR 格式）
    img_bgr = cv2.imread(image_path)
    # 转为 RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # 缩放至 224x224，使用 INTER_LINEAR（对应 Android 的 BILINEAR）
    resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    # 转为 float 并归一化到 [0,1]
    float_img = resized.astype(np.float32) / 255.0
    # 应用 ImageNet 归一化 (HWC -> CHW)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (float_img - mean) / std
    # 转为 CHW 并增加 batch 维度，并显式转为 float32（避免类型错误）
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).float()
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
    return probs

# --- 运行两种方式 ---
classes = ['细菌性斑点病', '早疫病', '晚疫病', '叶霉病', '斑枯病',
           '蜘蛛螨', '靶斑病', '黄曲叶病毒', '花叶病毒', '健康']

print("=== Web 版 (PIL+LANCZOS) ===")
probs_web = predict_web()
for i, p in enumerate(probs_web):
    print(f"{classes[i]}: {p.item()*100:.2f}%")
print(f"预测: {classes[probs_web.argmax()]} ({probs_web.max().item()*100:.2f}%)\n")

print("=== Android 模拟 (OpenCV BILINEAR) ===")
probs_android = predict_android_sim()
for i, p in enumerate(probs_android):
    print(f"{classes[i]}: {p.item()*100:.2f}%")
print(f"预测: {classes[probs_android.argmax()]} ({probs_android.max().item()*100:.2f}%)")