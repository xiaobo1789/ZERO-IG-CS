import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as st
from utils import pair_downsampler, calculate_local_variance, LocalMean  # 导入工具函数

EPS = 1e-9  # 防止除零
PI = 22.0 / 7.0  # 圆周率近似值


class LossFunction(nn.Module):
    # 总损失函数（组合多种损失项）
    def __init__(self):
        super(LossFunction, self).__init__()
        self._l2_loss = nn.MSELoss()  # L2损失（均方误差）
        self._l1_loss = nn.L1Loss()   # L1损失（平均绝对误差）
        self.smooth_loss = SmoothLoss()  # 平滑损失（保持空间连续性）
        self.texture_difference = TextureDifference()  # 纹理差异损失
        self.local_mean = LocalMean(patch_size=5)  # 局部均值计算
        self.L_TV_loss = L_TV()  # TV损失（总变分，保持平滑）


    def forward(self, input, L_pred1, L_pred2, L2, s2, s21, s22, H2, H11, H12, H13, s13, H14, s14, H3, s3, H3_pred, H4_pred, L_pred1_L_pred2_diff, H3_denoised1_H3_denoised2_diff, H2_blur, H3_blur):
        # 计算总损失（组合多个损失项）
        eps = 1e-9
        input = input + eps  # 输入加小值，避免除零

        # 计算输入图像的亮度（Y通道）
        input_Y = L2.detach()[:, 2, :, :] * 0.299 + L2.detach()[:, 1, :, :] * 0.587 + L2.detach()[:, 0, :, :] * 0.144
        input_Y_mean = torch.mean(input_Y, dim=(1, 2))  # 平均亮度
        # 计算增强因子（基于平均亮度，暗处增强更明显）
        enhancement_factor = 0.5 / (input_Y_mean + eps)
        enhancement_factor = enhancement_factor.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # 扩展维度匹配图像
        enhancement_factor = torch.clamp(enhancement_factor, 1, 25)  # 限制增强范围
        # 调整比率（防止过增强）
        adjustment_ratio = torch.pow(0.7, -enhancement_factor) / enhancement_factor
        adjustment_ratio = adjustment_ratio.repeat(1, 3, 1, 1)  # 扩展到3通道

        # 归一化低光层和增强亮度目标
        normalized_low_light_layer = L2.detach() / s2
        normalized_low_light_layer = torch.clamp(normalized_low_light_layer, eps, 0.8)
        enhanced_brightness = torch.pow(L2.detach() * enhancement_factor, enhancement_factor)
        clamped_enhanced_brightness = torch.clamp(enhanced_brightness * adjustment_ratio, eps, 1)
        clamped_adjusted_low_light = torch.clamp(L2.detach() * enhancement_factor, eps, 1)

        loss = 0  # 总损失初始化

        # 增强损失（约束光照图s2）
        loss += self._l2_loss(s2, clamped_enhanced_brightness) * 700  # 光照图与目标亮度的L2损失（权重700）
        loss += self._l2_loss(normalized_low_light_layer, clamped_adjusted_low_light) * 1000  # 归一化低光层损失（权重1000）
        loss += self.smooth_loss(L2.detach(), s2) * 5  # 平滑损失（权重5）
        loss += self.L_TV_loss(s2) * 1600  # TV损失（权重1600）

        # 残差一致性损失1（多尺度去噪一致性）
        L11, L12 = pair_downsampler(input)
        loss += self._l2_loss(L11, L_pred2) * 1000  # 下采样图与去噪结果的一致性（权重1000）
        loss += self._l2_loss(L12, L_pred1) * 1000
        denoised1, denoised2 = pair_downsampler(L2)
        loss += self._l2_loss(L_pred1, denoised1) * 1000  # 去噪结果与下采样去噪的一致性
        loss += self._l2_loss(L_pred2, denoised2) * 1000

        # 残差一致性损失2（第二级去噪一致性）
        loss += self._l2_loss(H3_pred, torch.cat([H12.detach(), s22.detach()], 1)) * 1000  # 去噪预测与输入的一致性
        loss += self._l2_loss(H4_pred, torch.cat([H11.detach(), s21.detach()], 1)) * 1000
        H3_denoised1, H3_denoised2 = pair_downsampler(H3)
        loss += self._l2_loss(H3_pred[:, 0:3, :, :], H3_denoised1) * 1000  # 去噪结果与下采样的一致性
        loss += self._l2_loss(H4_pred[:, 0:3, :, :], H3_denoised2) * 1000

        # 颜色一致性损失（模糊后颜色一致）
        loss += self._l2_loss(H2_blur.detach(), H3_blur) * 10000  # 权重10000

        # 光照一致性损失（光照图的一致性）
        loss += self._l2_loss(s2.detach(), s3) * 1000  # 权重1000

        # 内容一致性损失（局部均值约束）
        local_mean1 = self.local_mean(H3_denoised1)
        local_mean2 = self.local_mean(H3_denoised2)
        weighted_diff1 = (1 - H3_denoised1_H3_denoised2_diff) * local_mean1 + H3_denoised1 * H3_denoised1_H3_denoised2_diff
        weighted_diff2 = (1 - H3_denoised1_H3_denoised2_diff) * local_mean2 + H3_denoised1 * H3_denoised1_H3_denoised2_diff
        loss += self._l2_loss(H3_denoised1, weighted_diff1) * 10000  # 权重10000
        loss += self._l2_loss(H3_denoised2, weighted_diff2) * 10000

        # 方差一致性损失（噪声方差约束）
        noise_std = calculate_local_variance(H3 - H2)  # 噪声方差
        H2_var = calculate_local_variance(H2)  # 图像方差
        loss += self._l2_loss(H2_var, noise_std) * 1000  # 权重1000

        return loss


def local_mean(self, image):
    # 计算局部均值（同utils中的LocalMean.forward，此处可能为冗余定义）
    padding = self.patch_size // 2
    image = F.pad(image, (padding, padding, padding, padding), mode='reflect')
    patches = image.unfold(2, self.patch_size, 1).unfold(3, self.patch_size, 1)
    return patches.mean(dim=(4, 5))


def gauss_kernel(kernlen=21, nsig=3, channels=1):
    # 生成高斯核（numpy版本，用于初始化）
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))  # 1D高斯核
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))  # 2D高斯核
    kernel = kernel_raw / kernel_raw.sum()  # 归一化
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis=2)  # 按通道复制
    return out_filter


class TextureDifference(nn.Module):
    # 计算两图像的纹理差异（用于约束纹理一致性）
    def __init__(self, patch_size=5, constant_C=1e-5, threshold=0.975):
        super(TextureDifference, self).__init__()
        self.patch_size = patch_size  # 局部窗口大小
        self.constant_C = constant_C  # 防止除零的小值
        self.threshold = threshold  # 纹理相似性阈值

    def forward(self, image1, image2):
        # 转换为灰度图（简化纹理计算）
        image1 = self.rgb_to_gray(image1)
        image2 = self.rgb_to_gray(image2)

        # 计算局部标准差（反映纹理变化）
        stddev1 = self.local_stddev(image1)
        stddev2 = self.local_stddev(image2)
        # 计算纹理相似度（基于标准差）
        numerator = 2 * stddev1 * stddev2
        denominator = stddev1 ** 2 + stddev2 ** 2 + self.constant_C
        diff = numerator / denominator  # 范围[0,1]，1表示纹理完全一致

        # 应用阈值：超过阈值的视为纹理一致（1），否则不一致（0）
        binary_diff = torch.where(diff > self.threshold, torch.tensor(1.0, device=diff.device),
                                  torch.tensor(0.0, device=diff.device))
        return binary_diff

    def local_stddev(self, image):
        # 计算局部标准差
        padding = self.patch_size // 2
        image = F.pad(image, (padding, padding, padding, padding), mode='reflect')  # 填充
        patches = image.unfold(2, self.patch_size, 1).unfold(3, self.patch_size, 1)  # 提取窗口
        mean = patches.mean(dim=(4, 5), keepdim=True)  # 窗口均值
        squared_diff = (patches - mean) ** 2  # 平方差
        local_variance = squared_diff.mean(dim=(4, 5))  # 局部方差
        local_stddev = torch.sqrt(local_variance + 1e-9)  # 局部标准差
        return local_stddev

    def rgb_to_gray(self, image):
        # RGB转灰度（使用 luminance 公式）
        gray_image = 0.144 * image[:, 0, :, :] + 0.5870 * image[:, 1, :, :] + 0.299 * image[:, 2, :, :]
        return gray_image.unsqueeze(1)  # 增加通道维度


class L_TV(nn.Module):
    # TV损失（总变分损失，用于保持图像平滑，减少噪声）
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight  # 权重

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]  # 高度
        w_x = x.size()[3]  # 宽度
        # 计算水平和垂直方向的像素数（用于归一化）
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        # 水平方向差异的平方和（相邻像素）
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        # 垂直方向差异的平方和
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        # 总损失（加权平均）
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class Blur(nn.Module):
    # 模糊模块（使用高斯核）
    def __init__(self, nc):
        super(Blur, self).__init__()
        self.nc = nc  # 通道数
        # 生成高斯核并转为张量
        kernel = gauss_kernel(kernlen=21, nsig=3, channels=self.nc)
        kernel = torch.from_numpy(kernel).permute(2, 3, 0, 1).cuda()
        self.weight = nn.Parameter(data=kernel, requires_grad=False).cuda()  # 固定高斯核（不参与训练）

    def forward(self, x):
        # 检查输入通道数是否匹配
        if x.size(1) != self.nc:
            raise RuntimeError(
                "输入通道数[%d]与预设通道数[%d]不匹配" % (x.size(1), self.nc))
        # 卷积模糊
        x = F.conv2d(x, self.weight, stride=1, padding=10, groups=self.nc)
        return x


class SmoothLoss(nn.Module):
    # 平滑损失（约束光照图的空间平滑性，基于输入图像的颜色相似性）
    def __init__(self):
        super(SmoothLoss, self).__init__()
        self.sigma = 10  # 控制颜色相似性的衰减速度

    def rgb2yCbCr(self, input_im):
        # RGB转YCbCr颜色空间（用于提取亮度通道）
        im_flat = input_im.contiguous().view(-1, 3).float()  # 展平为(W*H,3)
        # YCbCr转换矩阵
        mat = torch.Tensor([[0.257, -0.148, 0.439], [0.564, -0.291, -0.368], [0.098, 0.439, -0.071]]).cuda()
        bias = torch.Tensor([16.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0]).cuda()  # 偏移量（归一化到[0,1]）
        temp = im_flat.mm(mat) + bias  # 矩阵乘法+偏移
        out = temp.view(input_im.shape[0], 3, input_im.shape[2], input_im.shape[3])  # 恢复形状(B,3,H,W)
        return out

    def forward(self, input, output):
        # 输入：原始图像；输出：光照图s2
        self.output = output
        self.input = self.rgb2yCbCr(input)  # 转换为YCbCr
        sigma_color = -1.0 / (2 * self.sigma * self.sigma)  # 颜色相似度衰减系数

        # 计算不同方向的颜色相似度权重（基于输入图像的颜色差异）
        # 水平方向（右→左）
        w1 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :] - self.input[:, :, :-1, :], 2), dim=1, keepdim=True) * sigma_color)
        w2 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :] - self.input[:, :, 1:, :], 2), dim=1, keepdim=True) * sigma_color)
        # 垂直方向（下→上）
        w3 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, 1:] - self.input[:, :, :, :-1], 2), dim=1, keepdim=True) * sigma_color)
        w4 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, :-1] - self.input[:, :, :, 1:], 2), dim=1, keepdim=True) * sigma_color)
        # 对角线方向（右下→左上）
        w5 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :-1] - self.input[:, :, 1:, 1:], 2), dim=1, keepdim=True) * sigma_color)
        w6 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, 1:] - self.input[:, :, :-1, :-1], 2), dim=1, keepdim=True) * sigma_color)
        # 反对角线方向（左下→右上）
        w7 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :-1] - self.input[:, :, :-1, 1:], 2), dim=1, keepdim=True) * sigma_color)
        w8 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, 1:] - self.input[:, :, 1:, :-1], 2), dim=1, keepdim=True) * sigma_color)
        # 2像素水平方向
        w9 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :] - self.input[:, :, :-2, :], 2), dim=1, keepdim=True) * sigma_color)
        w10 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :] - self.input[:, :, 2:, :], 2), dim=1, keepdim=True) * sigma_color)
        # 2像素垂直方向
        w11 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, 2:] - self.input[:, :, :, :-2], 2), dim=1, keepdim=True) * sigma_color)
        w12 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, :-2] - self.input[:, :, :, 2:], 2), dim=1, keepdim=True) * sigma_color)
        # 更多方向的权重（共24个方向，确保光照图在所有方向平滑）
        w13 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :-1] - self.input[:, :, 2:, 1:], 2), dim=1, keepdim=True) * sigma_color)
        w14 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, 1:] - self.input[:, :, :-2, :-1], 2), dim=1, keepdim=True) * sigma_color)
        w15 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :-1] - self.input[:, :, :-2, 1:], 2), dim=1, keepdim=True) * sigma_color)
        w16 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, 1:] - self.input[:, :, 2:, :-1], 2), dim=1, keepdim=True) * sigma_color)
        w17 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :-2] - self.input[:, :, 1:, 2:], 2), dim=1, keepdim=True) * sigma_color)
        w18 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, 2:] - self.input[:, :, :-1, :-2], 2), dim=1, keepdim=True) * sigma_color)
        w19 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :-2] - self.input[:, :, :-1, 2:], 2), dim=1, keepdim=True) * sigma_color)
        w20 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, 2:] - self.input[:, :, 1:, :-2], 2), dim=1, keepdim=True) * sigma_color)
        w21 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :-2] - self.input[:, :, 2:, 2:], 2), dim=1, keepdim=True) * sigma_color)
        w22 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, 2:] - self.input[:, :, :-2, :-2], 2), dim=1, keepdim=True) * sigma_color)
        w23 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :-2] - self.input[:, :, :-2, 2:], 2), dim=1, keepdim=True) * sigma_color)
        w24 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, 2:] - self.input[:, :, 2:, :-2], 2), dim=1, keepdim=True) * sigma_color)

        p = 1.0  # L1范数（控制平滑程度）

        # 计算光照图在各方向的差异（加权）
        pixel_grad1 = w1 * torch.norm((self.output[:, :, 1:, :] - self.output[:, :, :-1, :]), p, dim=1, keepdim=True)
        pixel_grad2 = w2 * torch.norm((self.output[:, :, :-1, :] - self.output[:, :, 1:, :]), p, dim=1, keepdim=True)
        pixel_grad3 = w3 * torch.norm((self.output[:, :, :, 1:] - self.output[:, :, :, :-1]), p, dim=1, keepdim=True)
        pixel_grad4 = w4 * torch.norm((self.output[:, :, :, :-1] - self.output[:, :, :, 1:]), p, dim=1, keepdim=True)
        pixel_grad5 = w5 * torch.norm((self.output[:, :, :-1, :-1] - self.output[:, :, 1:, 1:]), p, dim=1, keepdim=True)
        pixel_grad6 = w6 * torch.norm((self.output[:, :, 1:, 1:] - self.output[:, :, :-1, :-1]), p, dim=1, keepdim=True)
        pixel_grad7 = w7 * torch.norm((self.output[:, :, 1:, :-1] - self.output[:, :, :-1, 1:]), p, dim=1, keepdim=True)
        pixel_grad8 = w8 * torch.norm((self.output[:, :, :-1, 1:] - self.output[:, :, 1:, :-1]), p, dim=1, keepdim=True)
        pixel_grad9 = w9 * torch.norm((self.output[:, :, 2:, :] - self.output[:, :, :-2, :]), p, dim=1, keepdim=True)
        pixel_grad10 = w10 * torch.norm((self.output[:, :, :-2, :] - self.output[:, :, 2:, :]), p, dim=1, keepdim=True)
        pixel_grad11 = w11 * torch.norm((self.output[:, :, :, 2:] - self.output[:, :, :, :-2]), p, dim=1, keepdim=True)
        pixel_grad12 = w12 * torch.norm((self.output[:, :, :, :-2] - self.output[:, :, :, 2:]), p, dim=1, keepdim=True)
        pixel_grad13 = w13 * torch.norm((self.output[:, :, :-2, :-1] - self.output[:, :, 2:, 1:]), p, dim=1, keepdim=True)
        pixel_grad14 = w14 * torch.norm((self.output[:, :, 2:, 1:] - self.output[:, :, :-2, :-1]), p, dim=1, keepdim=True)
        pixel_grad15 = w15 * torch.norm((self.output[:, :, 2:, :-1] - self.output[:, :, :-2, 1:]), p, dim=1, keepdim=True)
        pixel_grad16 = w16 * torch.norm((self.output[:, :, :-2, 1:] - self.output[:, :, 2:, :-1]), p, dim=1, keepdim=True)
        pixel_grad17 = w17 * torch.norm((self.output[:, :, :-1, :-2] - self.output[:, :, 1:, 2:]), p, dim=1, keepdim=True)
        pixel_grad18 = w18 * torch.norm((self.output[:, :, 1:, 2:] - self.output[:, :, :-1, :-2]), p, dim=1, keepdim=True)
        pixel_grad19 = w19 * torch.norm((self.output[:, :, 1:, :-2] - self.output[:, :, :-1, 2:]), p, dim=1, keepdim=True)
        pixel_grad20 = w20 * torch.norm((self.output[:, :, :-1, 2:] - self.output[:, :, 1:, :-2]), p, dim=1, keepdim=True)
        pixel_grad21 = w21 * torch.norm((self.output[:, :, :-2, :-2] - self.output[:, :, 2:, 2:]), p, dim=1, keepdim=True)
        pixel_grad22 = w22 * torch.norm((self.output[:, :, 2:, 2:] - self.output[:, :, :-2, :-2]), p, dim=1, keepdim=True)
        pixel_grad23 = w23 * torch.norm((self.output[:, :, 2:, :-2] - self.output[:, :, :-2, 2:]), p, dim=1, keepdim=True)
        pixel_grad24 = w24 * torch.norm((self.output[:, :, :-2, 2:] - self.output[:, :, 2:, :-2]), p, dim=1, keepdim=True)

        # 总平滑损失（所有方向差异的均值和）
        ReguTerm1 = torch.mean(pixel_grad1) \
                    + torch.mean(pixel_grad2) \
                    + torch.mean(pixel_grad3) \
                    + torch.mean(pixel_grad4) \
                    + torch.mean(pixel_grad5) \
                    + torch.mean(pixel_grad6) \
                    + torch.mean(pixel_grad7) \
                    + torch.mean(pixel_grad8) \
                    + torch.mean(pixel_grad9) \
                    + torch.mean(pixel_grad10) \
                    + torch.mean(pixel_grad11) \
                    + torch.mean(pixel_grad12) \
                    + torch.mean(pixel_grad13) \
                    + torch.mean(pixel_grad14) \
                    + torch.mean(pixel_grad15) \
                    + torch.mean(pixel_grad16) \
                    + torch.mean(pixel_grad17) \
                    + torch.mean(pixel_grad18) \
                    + torch.mean(pixel_grad19) \
                    + torch.mean(pixel_grad20) \
                    + torch.mean(pixel_grad21) \
                    + torch.mean(pixel_grad22) \
                    + torch.mean(pixel_grad23) \
                    + torch.mean(pixel_grad24)

        total_term = ReguTerm1
        return total_term