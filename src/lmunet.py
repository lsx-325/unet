from typing import Dict  # 导入 Dict 类型提示
import torch  # 导入 PyTorch 核心库
import torch.nn as nn  # 导入 PyTorch 神经网络模块
import torch.nn.functional as F  # 导入 PyTorch 神经网络函数


# --------------------------
# 1. 基础模块: ConvBNReLU6 - 核心修改为 ReLU6
# --------------------------

class ConvBNReLU6(nn.Sequential):
    """
    一个结合了卷积、批量归一化和 ReLU6 激活的块。
    FPGA 优化点：
    1. 使用 ReLU6，其激活范围限制在 [0, 6]，对 8-bit 定点量化非常友好。
    2. 结构上便于推理时将 BN 参数融合到 Conv 参数中。
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        # 初始化 nn.Sequential 容器
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                         bias=False)  # Conv：不使用 Bias
        bn = nn.BatchNorm2d(out_channels)  # BN：标准批量归一化层
        # 关键修改：使用 ReLU6，支持量化友好
        relu6 = nn.ReLU6(inplace=True)

        # 明确地使用 Sequential 将三层按顺序打包
        super(ConvBNReLU6, self).__init__(conv, bn, relu6)


# --------------------------
# 2. 通道混洗 (Channel Shuffle)
# --------------------------

class ChannelShuffle(nn.Module):
    """
    通道混洗操作，用于增强特征通信，同时保持参数量极小。
    FPGA 友好性：实现为简单的片上数据交换（Transposition + Reshaping），硬件开销极小。
    """

    def __init__(self, channels, groups):
        super(ChannelShuffle, self).__init__()
        # 检查通道数是否能被组数整除
        if channels % groups != 0:
            raise ValueError("channels must be divisible by groups")
        self.groups = groups  # 记录分组数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 获取输入张量的尺寸信息 (Batch, Channels, Height, Width)
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups  # 计算每组的通道数

        # 1. Reshape: 将通道维度拆分成 (groups, channels_per_group)
        x = x.view(batchsize, self.groups, channels_per_group, height, width)

        # 2. Transpose (混洗的核心): 交换第 1 和 第 2 维度
        x = torch.transpose(x, 1, 2).contiguous()  # contiguous() 保证内存连续

        # 3. Flatten: 将通道相关维度合并回去
        x = x.view(batchsize, num_channels, height, width)
        return x


# --------------------------
# 3. MIR_FPGA - 规整化倒残差模块 (使用 ReLU6 和 Shuffle)
# --------------------------

class MIR_FPGA(nn.Module):
    """
    LMUNet-FPGA 的规整化倒残差 (Inverted Residual) 模块。
    FPGA 优化点：使用 ReLU6 (量化友好)、消除多尺度分支 (规整性)、引入 ChannelShuffle (通信效率)。
    """

    def __init__(self, in_channels, shuffle_groups=4):
        super(MIR_FPGA, self).__init__()

        expanded_channels = in_channels * 4  # 扩张通道，通常为 4 倍

        # 1x1 扩张卷积 (Pointwise): 提升通道数，使用 ConvBNReLU6
        self.expand_pw = ConvBNReLU6(in_channels, expanded_channels, kernel_size=1, padding=0)

        # 核心深度可分离卷积 (Depthwise): 提取空间特征
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, padding=1,
                      groups=expanded_channels, bias=False),  # DWC
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True)  # DWC 后使用 ReLU6
        )

        # 引入 Channel Shuffle
        self.shuffle = ChannelShuffle(expanded_channels, shuffle_groups)

        # 1x1 投影卷积 (Pointwise): 降低通道数，使用 ConvBNReLU6
        self.project_pw = ConvBNReLU6(expanded_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x  # 记录残差连接的输入

        # 1. 扩张
        x_expanded = self.expand_pw(x)
        # 2. 深度卷积
        x_dwc = self.depthwise_conv(x_expanded)
        # 3. 混洗
        x_shuffled = self.shuffle(x_dwc)
        # 4. 投影
        projected = self.project_pw(x_shuffled)

        # 5. 倒残差连接: 元素级相加
        return projected + residual


# --------------------------
# 4. AASPP_FPGA - 瓶颈层简化 (使用 ReLU6 和加法融合)
# --------------------------

class AASPP_FPGA(nn.Module):
    """
    LMUNet-FPGA 的规整化 AASPP 模块 (瓶颈层)。
    FPGA 优化点：用规整 DWC 取代非对称卷积，用加法融合特征，使用 ReLU6 (量化友好)。
    """

    def __init__(self, in_channels):
        super(AASPP_FPGA, self).__init__()
        projected_channels = in_channels // 2  # 初始投影通道数减半
        # 初始投影，使用 ConvBNReLU6
        self.initial_project = ConvBNReLU6(in_channels, projected_channels, kernel_size=1, padding=0)

        # 规整化空洞分支，使用不同的空洞率 (1, 2, 4)
        self.branch1 = self._make_regular_branch(projected_channels, dilation=1)
        self.branch2 = self._make_regular_branch(projected_channels, dilation=2)
        self.branch3 = self._make_regular_branch(projected_channels, dilation=4)

        # 最终投影，使用 ConvBNReLU6
        self.final_project = ConvBNReLU6(projected_channels, in_channels, kernel_size=1, padding=0)

    def _make_regular_branch(self, channels, dilation):
        """
        创建一个规整的空洞深度可分离卷积分支。
        """
        return nn.Sequential(
            # 规整的 3x3 DWC，空洞率由 dilation 控制
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation,
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU6(inplace=True)  # 使用 ReLU6
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.initial_project(x)

        b1 = self.branch1(projected)
        b2 = self.branch2(projected)
        b3 = self.branch3(projected)

        # 关键修改: 用逐元素相加取代拼接，简化硬件控制逻辑
        fused = projected + b1 + b2 + b3

        return self.final_project(fused)


# --------------------------
# 5. Up_FPGA - 解码器上采样模块
# --------------------------

class Up_FPGA(nn.Module):
    """
    LMUNet-FPGA 的上采样块。
    FPGA 优化点：消除了原代码中复杂的 F.pad 逻辑，使用 ConvBNReLU6。
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(Up_FPGA, self).__init__()
        # ConvTranspose2d 用于上采样 (核大小 2, 步长 2)
        self.up = nn.ConvTranspose2d(in_channels, skip_channels, kernel_size=2, stride=2)

        # Convolutions after the additive skip connection
        self.conv = nn.Sequential(
            ConvBNReLU6(skip_channels, out_channels),  # 使用 ConvBNReLU6
            ConvBNReLU6(out_channels, out_channels)  # 使用 ConvBNReLU6
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)  # 执行上采样

        # 优化点：此处假设尺寸已对齐，移除 F.pad 逻辑。

        # Additive skip connection (加法跳跃连接)
        x = x1 + x2
        return self.conv(x)


# --------------------------
# 6. OutConv - 输出卷积层
# --------------------------

class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            # 1x1 卷积将最终特征图通道数映射到类别数
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


# --------------------------
# 7. LMUNet-FPGA 主体网络
# --------------------------

class LMUNet_FPGA(nn.Module):
    """
    针对 FPGA 部署优化后的 LMUNet 最终版本。
    所有模块均采用硬件友好的设计，使用 ReLU6 确保量化友好。
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super(LMUNet_FPGA, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # 编码器 (Encoder) - 使用 ConvBNReLU6
        self.in_conv = nn.Sequential(ConvBNReLU6(in_channels, 16), ConvBNReLU6(16, 16))

        # 降采样和特征提取块，使用 MIR_FPGA
        self.down1 = nn.Sequential(nn.MaxPool2d(2), self._make_mir_layer(16, 3), ConvBNReLU6(16, 32, 1, 0))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), self._make_mir_layer(32, 4), ConvBNReLU6(32, 64, 1, 0))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), self._make_mir_layer(64, 6), ConvBNReLU6(64, 128, 1, 0))
        self.down4 = nn.MaxPool2d(2)

        # 瓶颈层 (Bottleneck)
        self.bottleneck = AASPP_FPGA(128)

        # 解码器 (Decoder) - 使用 Up_FPGA
        self.up1 = Up_FPGA(128, 128, 64)
        self.up2 = Up_FPGA(64, 64, 32)
        self.up3 = Up_FPGA(32, 32, 16)
        self.up4 = Up_FPGA(16, 16, 8)

        # 输出层 (Output Layer)
        self.out_conv = OutConv(8, num_classes)

    def _make_mir_layer(self, channels, num_modules):
        # 辅助函数：堆叠指定数量的 MIR_FPGA 模块
        return nn.Sequential(*[MIR_FPGA(channels) for _ in range(num_modules)])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 编码器路径 - 储存跳跃连接
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 瓶颈层
        b = self.bottleneck(x5)

        # 解码器路径与加法跳跃连接
        d1 = self.up1(b, x4)  # b 加上 x4
        d2 = self.up2(d1, x3)  # d1 加上 x3
        d3 = self.up3(d2, x2)  # d2 加上 x2
        d4 = self.up4(d3, x1)  # d3 加上 x1

        # 输出层
        logits = self.out_conv(d4)

        return {"out": logits}