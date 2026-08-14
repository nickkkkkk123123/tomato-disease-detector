import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

# ========== 页面配置 ==========
st.set_page_config(
    page_title="番茄叶片病害识别",
    page_icon="🍅",
    layout="centered"
)

# ========== 标题 ==========
st.title("🍅 番茄叶片病害识别系统")
st.markdown("上传一张番茄叶片图片，系统将自动识别是否患病及病害类型。")

# ========== 类别列表（与训练一致） ==========
CLASSES = [
    "细菌性斑点病", "早疫病", "晚疫病", "叶霉病", "斑枯病",
    "蜘蛛螨", "靶斑病", "黄曲叶病毒", "花叶病毒", "健康"
]

# ========== 加载模型（缓存，只加载一次） ==========
@st.cache_resource
def load_model():
    model_path = "best_model_multiclass.pth"
    if not os.path.exists(model_path):
        st.error(f"❌ 模型文件不存在: {model_path}")
        return None
    model = torch.load(model_path, map_location="cpu")
    model.eval()
    return model

# ========== 图片预处理 ==========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ========== 预测函数 ==========
def predict(image):
    model = load_model()
    if model is None:
        return None, None
    
    # 预处理
    img_tensor = transform(image).unsqueeze(0)
    
    # 推理
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    
    # 获取最高概率
    max_prob, max_idx = torch.max(probs, dim=0)
    label = CLASSES[max_idx.item()]
    confidence = max_prob.item() * 100
    
    # 全部概率（用于显示）
    all_probs = {CLASSES[i]: probs[i].item() * 100 for i in range(len(CLASSES))}
    
    return label, confidence, all_probs

# ========== UI ==========
uploaded_file = st.file_uploader(
    "📤 请选择一张番茄叶片图片",
    type=["jpg", "jpeg", "png"],
    help="支持 JPG、JPEG、PNG 格式"
)

if uploaded_file is not None:
    # 显示图片
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 上传的图片", use_column_width=True)
    
    # 识别按钮
    if st.button("🔍 开始识别", type="primary"):
        with st.spinner("🧠 模型推理中，请稍候..."):
            label, confidence, all_probs = predict(image)
        
        if label is not None:
            # 显示结果
            st.success(f"✅ 预测结果：**{label}**（置信度：{confidence:.2f}%）")
            
            # 显示所有类别概率（柱状图）
            st.subheader("📊 所有类别概率分布")
            st.bar_chart(all_probs)
        else:
            st.error("❌ 模型加载失败，请检查模型文件是否存在")

# ========== 底部信息 ==========
st.markdown("---")
st.caption("基于 MobileNetV2 的番茄病害 10 分类识别系统 | 模型准确率 99.71%")