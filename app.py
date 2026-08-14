import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# ============================================================
# 1. 配置参数
# ============================================================
MODEL_PATH = r"C:\Users\NICK\Desktop\tomato_project\best_model_multiclass.pth"

# 中文类别名称（顺序必须与训练时的 categories 一致）
class_names = [
    '细菌性斑点病', '早疫病', '晚疫病', '叶霉病', '斑枯病',
    '蜘蛛螨', '靶斑病', '黄曲叶病毒', '花叶病毒', '健康'
]

# 置信度阈值：低于此值判定为不确定
CONF_THRESHOLD = 0.60
# Top-2 置信度差值阈值：若小于此值，说明模型不够确定
DIFF_THRESHOLD = 0.30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ============================================================
# 2. 加载模型
# ============================================================
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.classifier[1].in_features, len(class_names))
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model

print("⏳ 正在加载模型...")
model = load_model()
print("✅ 模型加载成功")

# ============================================================
# 3. 数据预处理
# ============================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ============================================================
# 4. 单张图片预测（含不确定性判断）
# ============================================================
def predict_single(img):
    img = img.convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top_probs, top_indices = torch.topk(probs, 2)
        top1_conf = top_probs[0].item()
        top2_conf = top_probs[1].item()
        diff = top1_conf - top2_conf
        pred_idx = top_indices[0].item()

        # 判断是否为番茄：如果 top1 置信度低于阈值 或 与第二高置信度差异小，则认为不确定
        if top1_conf < CONF_THRESHOLD or diff < DIFF_THRESHOLD:
            label = "不确定/非番茄"
            confidence = top1_conf
        else:
            label = class_names[pred_idx]
            confidence = top1_conf
        return label, confidence

# ============================================================
# 5. 批量预测
# ============================================================
def predict_batch(files):
    results = []
    for file_path in files:
        try:
            img = Image.open(file_path)
            label, conf = predict_single(img)
            if label == "不确定/非番茄":
                status = "⚠️ 无法识别，请上传清晰的番茄叶片"
            else:
                status = "✅"
            results.append([
                os.path.basename(file_path),
                label,
                f"{conf*100:.1f}%",
                status
            ])
        except Exception as e:
            results.append([
                os.path.basename(file_path),
                f"错误: {str(e)}",
                "-",
                "❌"
            ])
    return results

# ============================================================
# 6. 启动 Gradio 界面
# ============================================================
demo = gr.Interface(
    fn=predict_batch,
    inputs=gr.Files(label="上传一张或多张图片（请确保为番茄叶片）"),
    outputs=gr.Dataframe(
        headers=["文件名", "预测结果", "置信度", "状态"],
        label="批量预测结果"
    ),
    title="🍅 番茄叶片病害识别系统（多分类）",
    description="支持识别 10 类番茄状态：9 种病害 + 健康。\n💡 如果模型不确定或图片非番茄，会提示「无法识别」。"
)

demo.launch(share=True)  # 启用公网链接