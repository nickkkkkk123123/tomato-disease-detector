import os
import shutil
from sklearn.model_selection import train_test_split

# ========== 配置路径 ==========
source_dir = r"C:\Users\NICK\Desktop\1\archive\PlantVillage"
target_dir = r"C:\Users\NICK\Desktop\tomato_multiclass"   # 新目录，存放划分好的数据

# 只取番茄类别（以 "Tomato" 开头的文件夹）
all_categories = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]
tomato_categories = [cat for cat in all_categories if cat.startswith("Tomato")]
print(f"发现 {len(tomato_categories)} 个番茄类别：")
for cat in tomato_categories:
    print(f"  {cat}")

# 创建目标目录结构
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(target_dir, split), exist_ok=True)

# 划分每个类别
for category in tomato_categories:
    category_path = os.path.join(source_dir, category)
    images = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        print(f"⚠️ 类别 {category} 没有图片，跳过")
        continue

    # 70% 训练，15% 验证，15% 测试
    train_imgs, val_test_imgs = train_test_split(images, test_size=0.3, random_state=42)
    val_imgs, test_imgs = train_test_split(val_test_imgs, test_size=0.5, random_state=42)

    # 复制图片到对应文件夹
    for split, img_list in zip(['train', 'val', 'test'], [train_imgs, val_imgs, test_imgs]):
        split_category_dir = os.path.join(target_dir, split, category)
        os.makedirs(split_category_dir, exist_ok=True)
        for img in img_list:
            src = os.path.join(category_path, img)
            dst = os.path.join(split_category_dir, img)
            shutil.copy2(src, dst)  # 保留元数据

    print(f"✅ {category}: 训练 {len(train_imgs)} 张, 验证 {len(val_imgs)} 张, 测试 {len(test_imgs)} 张")

print("🎉 数据划分完成！")