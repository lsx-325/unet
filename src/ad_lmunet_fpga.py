from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------
# 1. 基础模块: ConvBNReLU6 - 核心修改为 ReLU6
# --------------------------

class ConvBNReLU6(nn.Sequential):
    """
    一个结合了卷积、批量归一化和 ReLU6 激活的块。
    FPGA 优化点：使用 ReLU6 (量化友好)，结构上便于 BN 融合。
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                         bias=False)  # Conv：不使用 Bias
        bn = nn.BatchNorm2d(out_channels)  # BN：标准批量归一化层
        relu6 = nn.ReLU6(inplace=True)  # ReLU6：支持量化友好

        super(ConvBNReLU6, self).__init__(conv, bn, relu6)


# --------------------------
# 2. 核心创新模块: Squeeze-and-Excitation (SE) Attention
# --------------------------

class SEAttention(nn.Module):
    """
    Squeeze-and-Excitation (SE) 通道注意力模块。
    FPGA 优化点：只使用 GAP, 1x1 卷积和 ReLU6，结构简单，计算量小。
    """

    def __init__(self, channel, reduction=4):
        super(SEAttention, self).__init__()
        # Squeeze 操作：全局平均池化 (GAP)，硬件友好
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Excitation 操作：两个 1x1 卷积实现通道交互
        self.fc = nn.Sequential(
            # 第一个 1x1 Conv：降维
            nn.Conv2d(channel, channel // reduction, kernel_size=1, padding=0, bias=False),
            nn.ReLU6(inplace=True),
            # 第二个 1x1 Conv：升维
            nn.Conv2d(channel // reduction, channel, kernel_size=1, padding=0, bias=False),
            # Sigmoid 应替换为 Hard Sigmoid (H-Sigmoid) 来保证最终硬件实现时的低资源消耗。
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        y = self.avg_pool(x)  # Squeeze: (B, C, 1, 1)
        y = self.fc(y)  # Excitation: (B, C, 1, 1)

        # Scale 操作：逐通道乘法
        return x * y.expand_as(x)


# --------------------------
# 3. 通道混洗 (Channel Shuffle)
# --------------------------

class ChannelShuffle(nn.Module):
    # (代码与原 lmunet_fpga.py 保持一致，用于增强特征通信)
    def __init__(self, channels, groups):
        super(ChannelShuffle, self).__init__()
        if channels % groups != 0:
            raise ValueError("channels must be divisible by groups")
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, num_channels, height, width)
        return x


# --------------------------
# 4. 编码器创新模块: AD-MIR_FPGA (引入 Attention)
# --------------------------

class AD_MIR_FPGA(nn.Module):
    """
    注意力驱动的多尺度倒残差模块 (Attention-Driven MIR)。
    FPGA 创新点：在倒残差结构中嵌入了 SEAttention，以微小代价提升特征表达能力。
    """

    def __init__(self, in_channels, shuffle_groups=4):
        super(AD_MIR_FPGA, self).__init__()

        expanded_channels = in_channels * 4  # 扩张通道

        # 1x1 扩张卷积 (Pointwise)
        self.expand_pw = ConvBNReLU6(in_channels, expanded_channels, kernel_size=1, padding=0)

        # 核心深度可分离卷积 (Depthwise)
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, padding=1,
                      groups=expanded_channels, bias=False),  # DWC
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True)
        )

        # 引入 Channel Shuffle
        self.shuffle = ChannelShuffle(expanded_channels, shuffle_groups)

        # --- 创新点：在 DWC 后引入 SE Attention ---
        self.se_attention = SEAttention(expanded_channels, reduction=4)

        # 1x1 投影卷积 (Pointwise)
        self.project_pw = ConvBNReLU6(expanded_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x_expanded = self.expand_pw(x)
        x_dwc = self.depthwise_conv(x_expanded)
        x_shuffled = self.shuffle(x_dwc)

        # 引入注意力
        x_attended = self.se_attention(x_shuffled)

        projected = self.project_pw(x_attended)

        # 倒残差连接: 元素级相加
        return projected + residual


# --------------------------
# 5. 瓶颈层创新模块: GCF_FPGA (Global Context Fusion)
# --------------------------

class GCF_FPGA(nn.Module):
    """
    全局上下文融合 (Global Context Fusion) 模块。
    FPGA 创新点：用此规整结构彻底取代复杂的 AASPP 模块（消除空洞卷积）。
    """

    def __init__(self, in_channels):
        super(GCF_FPGA, self).__init__()
        reduced_channels = in_channels // 4  # 进一步压缩通道，用于上下文特征

        # 1. Squeeze: 全局平均池化 (GAP)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 2. Excitation: 提取/转换上下文特征
        self.conv_guide = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, padding=0, bias=False),
            nn.ReLU6(inplace=True),
        )

        # 3. Fuse: 投影回原始通道数，准备融合
        self.conv_fuse = ConvBNReLU6(reduced_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # 1. 提取全局上下文 (B, C, 1, 1)
        y = self.avg_pool(x)

        # 2. 上下文特征转换 (B, C/4, 1, 1)
        y = self.conv_guide(y)

        # 3. 上采样至原始尺寸 (使用最近邻插值，最硬件友好)
        # 硬件友好性：最近邻插值在 FPGA 上实现为简单的数据复制，开销极小。
        y = F.interpolate(y, size=x.size()[-2:], mode='nearest')

        # 4. 投影回原始通道数 (B, C, H, W)
        y = self.conv_fuse(y)

        # 5. 融合: 元素级相加
        return residual + y


# --------------------------
# 6. Up_FPGA 和 OutConv 保持不变
# --------------------------

class Up_FPGA(nn.Module):
    """
    LMUNet-FPGA 的上采样块，使用 ConvBNReLU6，并依赖于裁剪/规整输入尺寸。
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(Up_FPGA, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, skip_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            ConvBNReLU6(skip_channels, out_channels),
            ConvBNReLU6(out_channels, out_channels)
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # 裁剪 x2，使其尺寸与 x1 匹配 (FPGA 友好修正)
        if x2.size()[2] != x1.size()[2] or x2.size()[3] != x1.size()[3]:
            x2 = x2[:, :, :x1.size()[2], :x1.size()[3]]

        x = x1 + x2
        return self.conv(x)


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


# --------------------------
# 7. AD-LMUNet-FPGA 主体网络 (引入 Width Multiplier)
# --------------------------

class AD_LMUNet_FPGA(nn.Module):
    """
    注意力驱动的 LMUNet-FPGA 最终版本 (AD-LMUNet-FPGA)。
    FPGA 创新点：集成 AD_MIR_FPGA 和 GCF_FPGA，并支持 Width Multiplier (alpha)。
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1, alpha: float = 1.0):
        super(AD_LMUNet_FPGA, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.alpha = alpha

        # 辅助函数：根据 alpha 调整通道数
        def scale(c):
            # 保证通道数是 4 的倍数且至少为 8 (通常需要对 FPGA 的 PE Array 对齐)
            return max(8, int((c * alpha + 3) // 4 * 4))

            # 编码器 (Encoder) - 使用 scale() 函数缩放所有通道数

        c16 = scale(16)
        c32 = scale(32)
        c64 = scale(64)
        c128 = scale(128)

        self.in_conv = nn.Sequential(ConvBNReLU6(in_channels, c16), ConvBNReLU6(c16, c16))

        # 降采样和特征提取块，使用 AD_MIR_FPGA
        self.down1 = nn.Sequential(nn.MaxPool2d(2), self._make_mir_layer(c16, 3), ConvBNReLU6(c16, c32, 1, 0))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), self._make_mir_layer(c32, 4), ConvBNReLU6(c32, c64, 1, 0))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), self._make_mir_layer(c64, 6), ConvBNReLU6(c64, c128, 1, 0))
        self.down4 = nn.MaxPool2d(2)

        # 瓶颈层 (Bottleneck) - 使用 GCF_FPGA
        self.bottleneck = GCF_FPGA(c128)

        # 解码器 (Decoder) - 通道数同样被缩放
        self.up1 = Up_FPGA(c128, c128, c64)
        self.up2 = Up_FPGA(c64, c64, c32)
        self.up3 = Up_FPGA(c32, c32, c16)
        self.up4 = Up_FPGA(c16, c16, scale(8))  # 最终输出前的通道

        # 输出层 (Output Layer)
        self.out_conv = OutConv(scale(8), num_classes)

    def _make_mir_layer(self, channels, num_modules):
        # 辅助函数：堆叠指定数量的 AD_MIR_FPGA 模块
        return nn.Sequential(*[AD_MIR_FPGA(channels) for _ in range(num_modules)])

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
        d1 = self.up1(b, x4)
        d2 = self.up2(d1, x3)
        d3 = self.up3(d2, x2)
        d4 = self.up4(d3, x1)

        # 输出层
        logits = self.out_conv(d4)

        return {"out": logits}