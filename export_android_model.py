import torch
import torch.nn as nn
from torchvision import models

# 1. 加载 FP32 模型（和训练时一致）
model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.classifier[1].in_features, 2)
)
model.load_state_dict(torch.load("best_model_MobileNetV2.pth", map_location="cpu"))
model.eval()

# 2. 创建示例输入（用于追踪）
example_input = torch.randn(1, 3, 224, 224)

# 3. 使用 torch.jit.trace 导出（更稳定）
traced_model = torch.jit.trace(model, example_input)

# 4. 保存为 .pt 文件
traced_model.save("model_mobilenetv2_fp32.pt")
print("✅ 导出成功：model_mobilenetv2_fp32.pt")