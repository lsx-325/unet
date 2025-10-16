import torch
import os
from src.unet import UNet
from src.ad_lmunet_fpga import AD_LMUNet_FPGA
from torch.quantization import QConfig, MinMaxObserver

def print_model_size(model, label):
    """一个辅助函数，用于保存模型权重、打印其大小并删除临时文件。"""
    temp_path = "temp_model_weights.pth"
    torch.save(model.state_dict(), temp_path)
    size_mb = os.path.getsize(temp_path) / 1e6
    print(f"{label} 大小: {size_mb:.2f} MB")
    os.remove(temp_path)
    return size_mb

# --- 配置 ---
NUM_CLASSES = 2  # 项目标准配置 (1个前景类 + 1个背景类)
IN_CHANNELS = 3

# --- 1. 原始 UNet ---
print("--- 正在计算原始 UNet 的大小 ---")
# UNet 的 base_c=32 是 train.py 中的默认配置
original_unet = UNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, base_c=32)
original_unet.eval()
size_unet = print_model_size(original_unet, "原始 UNet")

# --- 2. AD-LMUNet-FPGA (FP32) ---
print("\n--- 正在计算 AD-LMUNet-FPGA (FP32) 的大小 ---")
ad_lmunet_fp32 = AD_LMUNet_FPGA(num_classes=NUM_CLASSES)
ad_lmunet_fp32.eval()
size_ad_lmunet_fp32 = print_model_size(ad_lmunet_fp32, "AD-LMUNet-FPGA (FP32)")

# --- 3. AD-LMUNet-FPGA (Quantized INT8) ---
print("\n--- 正在计算 AD-LMUNet-FPGA (Quantized INT8) 的大小 ---")
# 为量化创建一个全新的模型实例
model_to_quantize = AD_LMUNet_FPGA(num_classes=NUM_CLASSES)
model_to_quantize.eval()

# 配置量化参数
model_to_quantize.qconfig = QConfig(
    activation=MinMaxObserver.with_args(dtype=torch.quint8),
    weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
)

# 准备并转换模型到其量化版本
model_prepared = torch.quantization.prepare(model_to_quantize)
# 注意：此处未进行校准，因为我们只关心由INT8数据类型决定的最终模型大小，
# 而这与具体的校准值无关。
model_int8 = torch.quantization.convert(model_prepared)
size_ad_lmunet_int8 = print_model_size(model_int8, "AD-LMUNet-FPGA (INT8)")

# --- 4. 最终总结 ---
print("\n" + "="*35)
print("           模型大小对比总结")
print("="*35)
print(f"原始 UNet:               {size_unet:.2f} MB")
print(f"AD-LMUNet-FPGA (FP32):   {size_ad_lmunet_fp32:.2f} MB")
print(f"AD-LMUNet-FPGA (INT8):   {size_ad_lmunet_int8:.2f} MB")
print("-"*35)
print(f"AD-LMUNet 比原始 UNet 轻量化了 {(1 - size_ad_lmunet_fp32 / size_unet) * 100:.2f}%。")
print(f"8位量化进一步将模型压缩了 {(1 - size_ad_lmunet_int8 / size_ad_lmunet_fp32) * 100:.2f}%。")
print("="*35)