import torch
import os

def export_mobile_model(model_path, output_path='model_multiclass.pt'):
    print("📦 加载模型...")
    model = torch.load(model_path, map_location='cpu')
    model.eval()
    
    # 创建示例输入
    example_input = torch.randn(1, 3, 224, 224)
    
    print("🔧 追踪模型...")
    try:
        # 尝试用 trace（速度快，适合固定结构）
        traced = torch.jit.trace(model, example_input)
    except Exception as e:
        print(f"⚠️ Trace 失败: {e}")
        print("切换到 script 模式...")
        traced = torch.jit.script(model)
    
    print("📱 导出移动端格式...")
    try:
        # 尝试使用移动端优化（PyTorch 1.9+）
        from torch.utils.mobile_optimizer import optimize_for_mobile
        optimized = optimize_for_mobile(traced)
        optimized._save_for_lite_interpreter(output_path)
    except (ImportError, AttributeError):
        # 降级方案：直接用 save_for_lite_interpreter
        traced._save_for_lite_interpreter(output_path)
    
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"✅ 导出成功！文件: {output_path}，大小: {size_mb:.2f} MB")

# 调用导出函数
export_mobile_model('best_model_multiclass.pth')