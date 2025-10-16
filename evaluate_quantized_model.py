import torch
from src.ad_lmunet_fpga import AD_LMUNet_FPGA  # 确保这里导入的是你刚刚修改过的模型文件
from my_dataset import DriveDataset
import transforms as T
import os
from torch.quantization import QConfig, MinMaxObserver

# 从您的项目中导入评估函数
from train_utils import evaluate

def print_model_size(model, label, path="temp.p"):
    """打印模型大小的辅助函数"""
    torch.save(model.state_dict(), path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"{label} Size (MB): {size_mb:.2f}")
    os.remove(path)
    return size_mb

# --- 配置参数 ---
weights_path = "./save_weights/best_model.pth"
quantized_weights_path = "save_weights/quantized_best_model.pth"
num_classes = 1 + 1
device = torch.device("cpu") # 在CPU上进行量化和评估

# --- 1. 加载并评估原始的 FP32 模型 ---
print("--- 正在评估原始 FP32 模型 ---")
model_fp32 = AD_LMUNet_FPGA(num_classes=num_classes)
checkpoint = torch.load(weights_path, map_location='cpu')
# 兼容 train.py 和 train_multi_GPU.py 保存的权重
model_fp32.load_state_dict(checkpoint.get('model', checkpoint))
model_fp32.eval()

# 准备数据加载器
mean = (0.709, 0.381, 0.224)
std = (0.127, 0.079, 0.043)
crop_size = 480
val_dataset = DriveDataset("./", train=False, transforms=T.Compose([
    T.CenterCrop(crop_size), T.ToTensor(), T.Normalize(mean=mean, std=std)]))
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10, collate_fn=val_dataset.collate_fn)

confmat_fp32, dice_fp32 = evaluate(model_fp32, val_loader, device=device, num_classes=num_classes)
print("\nFP32 模型评估结果:")
print(confmat_fp32)
print(f"Dice Coefficient: {dice_fp32:.4f}")
print("-" * 30)

# --- 2. 创建、校准并保存 INT8 模型 ---
print("\n--- 正在创建并校准 INT8 模型 ---")
# 创建一个全新的、待量化的模型实例
model_to_quantize = AD_LMUNet_FPGA(num_classes=num_classes)
model_to_quantize.load_state_dict(checkpoint.get('model', checkpoint))
model_to_quantize.eval()

# 设置量化配置
model_to_quantize.qconfig = QConfig(
    activation=MinMaxObserver.with_args(dtype=torch.quint8),
    weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
)

# 准备模型，插入观察者
model_prepared = torch.quantization.prepare(model_to_quantize)

# 用验证数据进行校准
print("正在校准模型...")
with torch.no_grad():
    for image, _ in val_loader:
        model_prepared(image)
print("校准完成。")

# 转换成最终的量化模型
model_int8 = torch.quantization.convert(model_prepared)
torch.save(model_int8.state_dict(), quantized_weights_path)
print(f"INT8 模型已保存至 {quantized_weights_path}")
print("-" * 30)

# --- 3. 加载并评估量化后的 INT8 模型 ---
print("\n--- 正在评估量化后的 INT8 模型 ---")
# 注意：这里我们直接使用上一步转换得到的 model_int8 进行评估
# 因为它已经包含了校准信息，并且是正确的量化模型结构
confmat_int8, dice_int8 = evaluate(model_int8, val_loader, device=device, num_classes=num_classes)
print("\nINT8 模型评估结果:")
print(confmat_int8)
print(f"Dice Coefficient: {dice_int8:.4f}")
print("-" * 30)

# --- 4. 最终结果总结 ---
print("\n--- 精度与性能比较总结 ---")
fp32_size = os.path.getsize(weights_path) / 1e6
int8_size = os.path.getsize(quantized_weights_path) / 1e6
print(f"模型大小 (FP32): {fp32_size:.2f} MB")
print(f"模型大小 (INT8): {int8_size:.2f} MB")
print(f"大小缩减率: {(1 - int8_size / fp32_size) * 100:.2f}%")
print("-" * 20)
print(f"Dice 系数 (FP32): {dice_fp32:.4f}")
print(f"Dice 系数 (INT8): {dice_int8:.4f}")
print(f"Dice 系数下降值: {dice_fp32 - dice_int8:.4f}")
print("--- 评估完成 ---")