import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

# 加载原始模型
model = torch.jit.load('model_multiclass.pt', map_location='cpu')
model.eval()

# 优化并保存
optimized = optimize_for_mobile(model)
optimized._save_for_lite_interpreter('model_multiclass_mobile.pt')
print("✅ 优化模型导出成功")