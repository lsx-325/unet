# # import torch
# # import torch.nn as nn
# # from typing import Dict
# #
# #
# # # (我们不再需要 ConvBNReLU6 辅助类，因为我们回到了 nn.Upsample)
# #
# # # --- 1. AxialDW 模块 ---
# # class AxialDW(nn.Module):
# #     def __init__(self, dim, mixer_kernel, dilation=1):
# #         super().__init__()
# #         h, w = mixer_kernel
# #         self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups=dim, dilation=dilation)
# #         self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups=dim, dilation=dilation)
# #
# #     def forward(self, x):
# #         x = x + self.dw_h(x) + self.dw_w(x)
# #         return x
# #
# #
# # # --- 2. EncoderBlock (使用 ReLU6) ---
# # class EncoderBlock(nn.Module):
# #     def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
# #         super().__init__()
# #         self.dw = AxialDW(in_c, mixer_kernel=(7, 7))
# #         self.bn = nn.BatchNorm2d(in_c)
# #         self.pw = nn.Conv2d(in_c, out_c, kernel_size=1)
# #         self.down = nn.MaxPool2d((2, 2))
# #         self.act = nn.ReLU6(inplace=True)  # 使用 ReLU6
# #
# #     def forward(self, x):
# #         skip = self.bn(self.dw(x))  #
# #         x = self.act(self.down(self.pw(skip)))
# #         return x, skip
# #
# #
# # # --- 3. DecoderBlock (使用 nn.Upsample) ---
# # class DecoderBlock(nn.Module):
# #     def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
# #         super().__init__()
# #
# #         # 恢复为 nn.Upsample
# #         self.up = nn.Upsample(scale_factor=2)
# #         self.pw = nn.Conv2d(in_c + out_c, out_c, kernel_size=1)
# #
# #         self.bn = nn.BatchNorm2d(out_c)
# #         self.dw = AxialDW(out_c, mixer_kernel=(7, 7))
# #         self.act = nn.ReLU6(inplace=True)  # 使用 ReLU6
# #         self.pw2 = nn.Conv2d(out_c, out_c, kernel_size=1)
# #
# #     def forward(self, x, skip):
# #         x = self.up(x)  # 使用 nn.Upsample
# #         x = torch.cat([x, skip], dim=1)  # , dim=1)]
# #         x = self.act(self.pw2(self.dw(self.bn(self.pw(x)))))
# #         return x
# #
# #
# # # --- 4. BottleNeckBlock (使用 ReLU6) ---
# # class BottleNeckBlock(nn.Module):
# #     def __init__(self, dim):
# #         super().__init__()
# #         gc = dim // 4
# #         self.pw1 = nn.Conv2d(dim, gc, kernel_size=1)
# #         self.dw1 = AxialDW(gc, mixer_kernel=(3, 3), dilation=1)
# #         self.dw2 = AxialDW(gc, mixer_kernel=(3, 3), dilation=2)
# #         self.dw3 = AxialDW(gc, mixer_kernel=(3, 3), dilation=3)
# #         self.bn = nn.BatchNorm2d(4 * gc)
# #         self.pw2 = nn.Conv2d(4 * gc, dim, kernel_size=1)
# #         self.act = nn.ReLU6(inplace=True)  # 使用 ReLU6
# #
# #     def forward(self, x):
# #         x = self.pw1(x)
# #         x = torch.cat([x, self.dw1(x), self.dw2(x), self.dw3(x)], 1)
# #         x = self.act(self.pw2(self.bn(x)))
# #         return x
# #
# #
# # # --- 5. ULite 主体 (关键修改：改为 4 层) ---
# # class ULite(nn.Module):
# #     def __init__(self, in_channels: int = 3, num_classes: int = 2):
# #         super().__init__()
# #
# #         """Encoder"""
# #         self.conv_in = nn.Conv2d(in_channels, 16, kernel_size=3, padding='same')  #
# #         self.e1 = EncoderBlock(16, 32)
# #         self.e2 = EncoderBlock(32, 64)
# #         self.e3 = EncoderBlock(64, 128)
# #         # --- [修改] 移除第 5 层 ---
# #         self.e4 = EncoderBlock(128, 256)
# #         # self.e5 = EncoderBlock(256, 512) # 移除
# #         # --- [结束] ---
# #
# #         """Bottle Neck"""
# #         # --- [修改] 瓶颈层现在在第 4 层之后 ---
# #         self.b4 = BottleNeckBlock(256)  # 之前是 self.b5(512)
# #         # --- [结束] ---
# #
# #         """Decoder"""
# #         # --- [修改] 移除第 5 层 ---
# #         # self.d5 = DecoderBlock(512, 256) # 移除
# #         self.d4 = DecoderBlock(256, 128)
# #         self.d3 = DecoderBlock(128, 64)
# #         self.d2 = DecoderBlock(64, 32)
# #         self.d1 = DecoderBlock(32, 16)
# #         self.conv_out = nn.Conv2d(16, num_classes, kernel_size=1)
# #         # --- [结束] ---
# #
# #     def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
# #         """Encoder"""
# #         x = self.conv_in(x)
# #         x, skip1 = self.e1(x)
# #         x, skip2 = self.e2(x)
# #         x, skip3 = self.e3(x)
# #         # --- [修改] ---
# #         x, skip4 = self.e4(x)
# #         # x, skip5 = self.e5(x) # 移除
# #         # --- [结束] ---
# #
# #         """BottleNeck"""
# #         # --- [修改] ---
# #         x = self.b4(x)
# #         # x = self.b5(x) # 移除
# #         # --- [结束] ---
# #
# #         """Decoder"""
# #         # --- [修改] ---
# #         # x = self.d5(x, skip5) # 移除
# #         x = self.d4(x, skip4)
# #         x = self.d3(x, skip3)
# #         x = self.d2(x, skip2)
# #         x = self.d1(x, skip1)
# #         x = self.conv_out(x)
# #         # --- [结束] ---
# #
# #         return {"out": x}
#
#
# # import torch
# # import torch.nn as nn
# # from typing import Dict
# #
# #
# # # (我们不再需要 ConvBNReLU6 辅助类，因为我们回到了 nn.Upsample)
# #
# # # --- 1. AxialDW 模块 (无变化) ---
# # class AxialDW(nn.Module):
# #     def __init__(self, dim, mixer_kernel, dilation=1):
# #         super().__init__()
# #         h, w = mixer_kernel
# #         self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups=dim, dilation=dilation)
# #         self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups=dim, dilation=dilation)
# #
# #     def forward(self, x):
# #         x = x + self.dw_h(x) + self.dw_w(x)
# #         return x
# #
# #
# # # --- [修改] 2. EncoderBlock (使用可学习的 Strided Conv) ---
# # # ----------------------------------------------------
# # class EncoderBlock(nn.Module):
# #     def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
# #         super().__init__()
# #         self.dw = AxialDW(in_c, mixer_kernel=(7, 7))
# #         self.bn = nn.BatchNorm2d(in_c)
# #
# #         # --- [修改] ---
# #         # 1. 移除 1x1 卷积 和 MaxPool2d
# #         # self.pw = nn.Conv2d(in_c, out_c, kernel_size=1)
# #         # self.down = nn.MaxPool2d((2, 2))
# #
# #         # 2. 替换为单个 3x3、stride=2 的可学习卷积
# #         #    它同时完成“改变通道”和“降低分辨率”
# #         self.pw_down = nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1)
# #         # --- [修改结束] ---
# #
# #         self.act = nn.ReLU6(inplace=True)  # 使用 ReLU6
# #
# #     def forward(self, x):
# #         skip = self.bn(self.dw(x))
# #
# #         # --- [修改] ---
# #         # 使用新的 pw_down 层进行下采样
# #         x = self.act(self.pw_down(skip))
# #         # --- [修改结束] ---
# #
# #         return x, skip
# #
# #
# # # --- 3. DecoderBlock (无变化, 保持 nn.Upsample) ---
# # class DecoderBlock(nn.Module):
# #     def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
# #         super().__init__()
# #
# #         # 保持 nn.Upsample (根据您的要求)
# #         self.up = nn.Upsample(scale_factor=2)
# #         self.pw = nn.Conv2d(in_c + out_c, out_c, kernel_size=1)
# #
# #         self.bn = nn.BatchNorm2d(out_c)
# #         self.dw = AxialDW(out_c, mixer_kernel=(7, 7))
# #         self.act = nn.ReLU6(inplace=True)  # 使用 ReLU6
# #         self.pw2 = nn.Conv2d(out_c, out_c, kernel_size=1)
# #
# #     def forward(self, x, skip):
# #         x = self.up(x)  # 使用 nn.Upsample
# #         x = torch.cat([x, skip], dim=1)
# #         x = self.act(self.pw2(self.dw(self.bn(self.pw(x)))))
# #         return x
# #
# #
# # # --- 4. BottleNeckBlock (无变化) ---
# # class BottleNeckBlock(nn.Module):
# #     def __init__(self, dim):
# #         super().__init__()
# #         gc = dim // 4
# #         self.pw1 = nn.Conv2d(dim, gc, kernel_size=1)
# #         self.dw1 = AxialDW(gc, mixer_kernel=(3, 3), dilation=1)
# #         self.dw2 = AxialDW(gc, mixer_kernel=(3, 3), dilation=2)
# #         self.dw3 = AxialDW(gc, mixer_kernel=(3, 3), dilation=3)
# #         self.bn = nn.BatchNorm2d(4 * gc)
# #         self.pw2 = nn.Conv2d(4 * gc, dim, kernel_size=1)
# #         self.act = nn.ReLU6(inplace=True)  # 使用 ReLU6
# #
# #     def forward(self, x):
# #         x = self.pw1(x)
# #         x = torch.cat([x, self.dw1(x), self.dw2(x), self.dw3(x)], 1)
# #         x = self.act(self.pw2(self.bn(x)))
# #         return x
# #
# #
# # # --- 5. ULite 主体 (无变化) ---
# # # (由于更改被封装在 EncoderBlock 中，这里无需任何改动)
# # class ULite(nn.Module):
# #     def __init__(self, in_channels: int = 3, num_classes: int = 2):
# #         super().__init__()
# #
# #         """Encoder"""
# #         self.conv_in = nn.Conv2d(in_channels, 16, kernel_size=3, padding='same')
# #         # e1(16, 32) -> skip1 是 16 通道
# #         self.e1 = EncoderBlock(16, 32)
# #         # e2(32, 64) -> skip2 是 32 通道
# #         self.e2 = EncoderBlock(32, 64)
# #         # e3(64, 128) -> skip3 是 64 通道
# #         self.e3 = EncoderBlock(64, 128)
# #         # e4(128, 256) -> skip4 是 128 通道
# #         self.e4 = EncoderBlock(128, 256)
# #
# #         """Bottle Neck"""
# #         self.b4 = BottleNeckBlock(256)
# #
# #         """Decoder"""
# #         # d4(256, 128) -> up(x) [B, 256] + skip4 [B, 128] -> cat [B, 384] -> pw(384, 128)
# #         self.d4 = DecoderBlock(256, 128)
# #         # d3(128, 64) -> up(x) [B, 128] + skip3 [B, 64] -> cat [B, 192] -> pw(192, 64)
# #         self.d3 = DecoderBlock(128, 64)
# #         # d2(64, 32) -> up(x) [B, 64] + skip2 [B, 32] -> cat [B, 96] -> pw(96, 32)
# #         self.d2 = DecoderBlock(64, 32)
# #         # d1(32, 16) -> up(x) [B, 32] + skip1 [B, 16] -> cat [B, 48] -> pw(48, 16)
# #         self.d1 = DecoderBlock(32, 16)
# #         self.conv_out = nn.Conv2d(16, num_classes, kernel_size=1)
# #
# #     def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
# #         """Encoder"""
# #         x = self.conv_in(x)
# #         x, skip1 = self.e1(x)
# #         x, skip2 = self.e2(x)
# #         x, skip3 = self.e3(x)
# #         x, skip4 = self.e4(x)
# #
# #         """BottleNeck"""
# #         x = self.b4(x)
# #
# #         """Decoder"""
# #         x = self.d4(x, skip4)
# #         x = self.d3(x, skip3)
# #         x = self.d2(x, skip2)
# #         x = self.d1(x, skip1)
# #         x = self.conv_out(x)
# #
# #         return {"out": x}
#
# import torch
# import torch.nn as nn
# from typing import Dict
#
#
# # --- 1. AxialDW 模块 (无变化) ---
# class AxialDW(nn.Module):
#     def __init__(self, dim, mixer_kernel, dilation=1):
#         super().__init__()
#         h, w = mixer_kernel
#         self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups=dim, dilation=dilation)
#         self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups=dim, dilation=dilation)
#
#     def forward(self, x):
#         x = x + self.dw_h(x) + self.dw_w(x)
#         return x
#
#
# # --- 2. EncoderBlock (保留 +0.2 提升的版本) ---
# class EncoderBlock(nn.Module):
#     def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
#         super().__init__()
#         self.dw = AxialDW(in_c, mixer_kernel=(7, 7))
#         self.bn = nn.BatchNorm2d(in_c)
#
#         # 保留这个可学习的下采样层
#         self.pw_down = nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1)
#         self.act = nn.ReLU6(inplace=True)
#
#     def forward(self, x):
#         skip = self.bn(self.dw(x))
#         x = self.act(self.pw_down(skip))
#         return x, skip
#
#
# # --- [重大修改] 3. DecoderBlock (集成 TransposedConv 和 MT-UNet 的残差) ---
# # -----------------------------------------------------------------
# class DecoderBlock(nn.Module):
#     def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
#         super().__init__()
#
#         # --- [修改 1: 来自 MT-UNet 的可学习上采样] ---
#         # 使用转置卷积替换 nn.Upsample
#         self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
#
#         # pw 的输入通道更新为 (out_c (来自 up) + out_c (来自 skip))
#         self.pw = nn.Conv2d(out_c + out_c, out_c, kernel_size=1)
#         # --- [修改 1 结束] ---
#
#         self.bn = nn.BatchNorm2d(out_c)
#         self.dw = AxialDW(out_c, mixer_kernel=(7, 7))
#         self.act = nn.ReLU6(inplace=True)
#         self.pw2 = nn.Conv2d(out_c, out_c, kernel_size=1)
#
#     def forward(self, x, skip):
#         x = self.up(x)  # <-- (1) 可学习的上采样
#         x = torch.cat([x, skip], dim=1)
#
#         # --- [修改 2: 来自 MT-UNet DoubleConv 的残差思想] ---
#         # h 是主干路径 (1x1 卷积)
#         h = self.pw(x)
#
#         # x_res 是残差路径 (DW + 1x1 卷积)
#         x_res = self.bn(h)
#         x_res = self.dw(x_res)
#         x_res = self.pw2(x_res)
#
#         # 相加 (h + x_res)，然后激活
#         x = self.act(h + x_res)
#         # --- [修改 2 结束] ---
#
#         return x
#
#
# # --- [修改结束] ---
#
#
# # --- 4. BottleNeckBlock (无变化) ---
# class BottleNeckBlock(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         gc = dim // 4
#         self.pw1 = nn.Conv2d(dim, gc, kernel_size=1)
#         self.dw1 = AxialDW(gc, mixer_kernel=(3, 3), dilation=1)
#         self.dw2 = AxialDW(gc, mixer_kernel=(3, 3), dilation=2)
#         self.dw3 = AxialDW(gc, mixer_kernel=(3, 3), dilation=3)
#         self.bn = nn.BatchNorm2d(4 * gc)
#         self.pw2 = nn.Conv2d(4 * gc, dim, kernel_size=1)
#         self.act = nn.ReLU6(inplace=True)
#
#     def forward(self, x):
#         x = self.pw1(x)
#         x = torch.cat([x, self.dw1(x), self.dw2(x), self.dw3(x)], 1)
#         x = self.act(self.pw2(self.bn(x)))
#         return x
#
#
# # --- 5. ULite 主体 (无变化) ---
# class ULite(nn.Module):
#     def __init__(self, in_channels: int = 3, num_classes: int = 2):
#         super().__init__()
#
#         """Encoder"""
#         self.conv_in = nn.Conv2d(in_channels, 16, kernel_size=3, padding='same')
#         self.e1 = EncoderBlock(16, 32)
#         self.e2 = EncoderBlock(32, 64)
#         self.e3 = EncoderBlock(64, 128)
#         self.e4 = EncoderBlock(128, 256)
#
#         """Bottle Neck"""
#         self.b4 = BottleNeckBlock(256)
#
#         """Decoder"""
#         # DecoderBlock 现在是我们的“终极”版本
#         self.d4 = DecoderBlock(256, 128)
#         self.d3 = DecoderBlock(128, 64)
#         self.d2 = DecoderBlock(64, 32)
#         self.d1 = DecoderBlock(32, 16)
#         self.conv_out = nn.Conv2d(16, num_classes, kernel_size=1)
#
#     def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
#         """Encoder"""
#         x = self.conv_in(x)
#         x, skip1 = self.e1(x)  # skip1 [16 ch]
#         x, skip2 = self.e2(x)  # skip2 [32 ch]
#         x, skip3 = self.e3(x)  # skip3 [64 ch]
#         x, skip4 = self.e4(x)  # skip4 [128 ch]
#
#         """BottleNeck"""
#         x = self.b4(x)  # x [256 ch]
#
#         """Decoder"""
#         x = self.d4(x, skip4)  # x [128 ch]
#         x = self.d3(x, skip3)  # x [64 ch]
#         x = self.d2(x, skip2)  # x [32 ch]
#         x = self.d1(x, skip1)  # x [16 ch]
#         x = self.conv_out(x)
#
#         return {"out": x}


import torch
import torch.nn as nn
from typing import Dict


# --- 1. AxialDW 模块 (无变化) ---
class AxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation=1):
        super().__init__()
        h, w = mixer_kernel
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups=dim, dilation=dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups=dim, dilation=dilation)

    def forward(self, x):
        x = x + self.dw_h(x) + self.dw_w(x)
        return x


# --- 2. EncoderBlock (恢复为原始的 MaxPool 版本) ---
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
        super().__init__()
        self.dw = AxialDW(in_c, mixer_kernel=(7, 7))
        self.bn = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1)  # 1x1 卷积
        self.down = nn.MaxPool2d((2, 2))  # 最大池化
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        skip = self.bn(self.dw(x))
        x = self.act(self.down(self.pw(skip)))  # 原始下采样路径
        return x, skip


# --- [修改] 3. DecoderBlock (原始 Upsample + 内部残差连接) ---
# --------------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
        super().__init__()

        # 恢复为 nn.Upsample
        self.up = nn.Upsample(scale_factor=2)
        self.pw = nn.Conv2d(in_c + out_c, out_c, kernel_size=1)

        self.bn = nn.BatchNorm2d(out_c)
        self.dw = AxialDW(out_c, mixer_kernel=(7, 7))
        self.act = nn.ReLU6(inplace=True)
        self.pw2 = nn.Conv2d(out_c, out_c, kernel_size=1)

    def forward(self, x, skip):
        x = self.up(x)  # 使用 nn.Upsample
        x = torch.cat([x, skip], dim=1)

        # --- [修改: 添加内部残差连接] ---
        # 1. h 作为残差连接的 "捷径" (Identity path)
        h = self.pw(x)

        # 2. x_res 作为残差路径 (Residual path)
        x_res = self.bn(h)
        x_res = self.dw(x_res)
        x_res = self.pw2(x_res)

        # 3. 相加，然后激活 (h + x_res)
        x = self.act(h + x_res)
        # --- [修改结束] ---

        return x


# --- 4. BottleNeckBlock (原始版本) ---
class BottleNeckBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        gc = dim // 4
        self.pw1 = nn.Conv2d(dim, gc, kernel_size=1)
        self.dw1 = AxialDW(gc, mixer_kernel=(3, 3), dilation=1)
        self.dw2 = AxialDW(gc, mixer_kernel=(3, 3), dilation=2)
        self.dw3 = AxialDW(gc, mixer_kernel=(3, 3), dilation=3)
        self.bn = nn.BatchNorm2d(4 * gc)
        self.pw2 = nn.Conv2d(4 * gc, dim, kernel_size=1)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.pw1(x)
        x = torch.cat([x, self.dw1(x), self.dw2(x), self.dw3(x)], 1)
        x = self.act(self.pw2(self.bn(x)))
        return x


# --- 5. ULite 主体 (原始版本) ---
class ULite(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()

        """Encoder"""
        self.conv_in = nn.Conv2d(in_channels, 16, kernel_size=3, padding='same')
        self.e1 = EncoderBlock(16, 32)
        self.e2 = EncoderBlock(32, 64)
        self.e3 = EncoderBlock(64, 128)
        self.e4 = EncoderBlock(128, 256)

        """Bottle Neck"""
        self.b4 = BottleNeckBlock(256)

        """Decoder"""
        # 这里将自动使用我们上面修改过的 DecoderBlock
        self.d4 = DecoderBlock(256, 128)
        self.d3 = DecoderBlock(128, 64)
        self.d2 = DecoderBlock(64, 32)
        self.d1 = DecoderBlock(32, 16)
        self.conv_out = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encoder"""
        x = self.conv_in(x)
        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        x, skip4 = self.e4(x)

        """BottleNeck"""
        x = self.b4(x)

        """Decoder"""
        x = self.d4(x, skip4)
        x = self.d3(x, skip3)
        x = self.d2(x, skip2)
        x = self.d1(x, skip1)
        x = self.conv_out(x)

        return {"out": x}