import torch
from src.ad_lmunet_fpga import AD_LMUNet_FPGA
from my_dataset import DriveDataset
import transforms as T
import os
# --- Start of fix ---
# Use the observers and QConfig available in PyTorch 1.13
from torch.quantization import QConfig, MinMaxObserver


# --- End of fix ---


def print_model_size(model, label):
    """打印模型大小的辅助函数"""
    torch.save(model.state_dict(), "temp.p")
    print(f"{label} Size (MB):", os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')


# 1. 加载你的预训练模型
weights_path = "./save_weights/best_model.pth"
num_classes = 1 + 1

model_fp32 = AD_LMUNet_FPGA(num_classes=num_classes)
checkpoint = torch.load(weights_path, map_location='cpu')

if 'model' in checkpoint:
    model_fp32.load_state_dict(checkpoint['model'])
else:
    model_fp32.load_state_dict(checkpoint)

model_fp32.eval()
print_model_size(model_fp32, "原始FP32模型")

# 2. 配置量化参数
# --- Start of fix ---
# For PyTorch versions < 1.13, you configure the observer directly
# to handle per-tensor quantization for weights.
model_fp32.qconfig = QConfig(
    activation=MinMaxObserver.with_args(dtype=torch.quint8),
    weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
)
# --- End of fix ---


# 3. 准备模型以进行静态量化
model_fp32_prepared = torch.quantization.prepare(model_fp32)

# 4. 校准模型
mean = (0.709, 0.381, 0.224)
std = (0.127, 0.079, 0.043)
crop_size = 480

val_dataset = DriveDataset("./",
                           train=False,
                           transforms=T.Compose([
                               T.CenterCrop(crop_size),
                               T.ToTensor(),
                               T.Normalize(mean=mean, std=std),
                           ]))

val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=10,
                                         collate_fn=val_dataset.collate_fn)

print("正在校准模型...")
with torch.no_grad():
    for image, _ in val_loader:
        model_fp32_prepared(image)
        break
print("校准完成。")

# 5. 转换成量化模型
model_int8 = torch.quantization.convert(model_fp32_prepared)
print_model_size(model_int8, "量化后INT8模型")

# 6. 保存量化后的模型
torch.save(model_int8.state_dict(), "save_weights/quantized_best_model.pth")
print("8位量化模型已保存至 save_weights/quantized_best_model.pth")