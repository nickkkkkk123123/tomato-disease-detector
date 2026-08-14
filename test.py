import torch
from PIL import Image
from torchvision import transforms

# 1. 加载模型
model = torch.jit.load('model_multiclass.pt', map_location='cpu')
model.eval()

# 2. 加载图片
img = Image.open(r'C:\Users\NICK\Desktop\54fbb2fb43166d22b8d888d8963bc0e79152d2fd.jpeg').convert('RGB')

# 3. 预处理（必须和 Android 端完全一致）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
input_tensor = transform(img).unsqueeze(0)

# 4. 推理
with torch.no_grad():
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)[0]

# 5. 显示结果
classes = ['细菌性斑点病', '早疫病', '晚疫病', '叶霉病', '斑枯病',
           '蜘蛛螨', '靶斑病', '黄曲叶病毒', '花叶病毒', '健康']

for i, prob in enumerate(probs):
    print(f"{classes[i]}: {prob.item()*100:.2f}%")