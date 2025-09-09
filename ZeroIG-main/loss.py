import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as st
import timm
from torchvision.models import vgg19, VGG19_Weights
from utils import pair_downsampler, calculate_local_variance, LocalMean, gauss_kernel  # 导入工具函数
from torch.nn.utils import spectral_norm
# 尝试导入LPIPS，如果不可用则回退到VGG
try:
    import lpips

    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("LPIPS not available, using VGG-based perceptual loss")

EPS = 1e-9  # 防止除零
PI = 22.0 / 7.0  # 圆周率近似值


# 多尺度SSIM损失
class MultiScaleSSIMLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights or [0.5, 0.3, 0.2]  # 多尺度权重

    def forward(self, pred, target):
        loss = 0
        for i, scale in enumerate([1.0, 0.5, 0.25]):  # 全尺度、半尺度、1/4尺度
            if scale != 1.0:
                pred_scale = F.interpolate(pred, scale_factor=scale, mode='bilinear')
                target_scale = F.interpolate(target, scale_factor=scale, mode='bilinear')
            else:
                pred_scale, target_scale = pred, target

            ssim_loss = 1 - self.ssim(pred_scale, target_scale)
            loss += ssim_loss * self.weights[i]
        return loss

    def ssim(self, pred, target, window_size=11, size_average=True):
        # 简化版SSIM实现
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu1 = F.avg_pool2d(pred, window_size, 1, 0)
        mu2 = F.avg_pool2d(target, window_size, 1, 0)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool2d(pred * pred, window_size, 1, 0) - mu1_sq
        sigma2_sq = F.avg_pool2d(target * target, window_size, 1, 0) - mu2_sq
        sigma12 = F.avg_pool2d(pred * target, window_size, 1, 0) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


# loss.py 中 ImprovedPerceptualLoss 类修改
class ImprovedPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lpips_available = LPIPS_AVAILABLE
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)  # 提前定义池化层
        if self.lpips_available:
            # LPIPS可用时，仅初始化LPIPS，延迟VGG19
            self.lpips = lpips.LPIPS(net='vgg')
            self.vgg = None
            self.slice1 = None
            self.slice2 = None
        else:
            # LPIPS不可用时，也延迟VGG19初始化
            self.lpips = None
            self.vgg = None
            self.slice1 = None
            self.slice2 = None

    def forward(self, pred, target):
        if self.lpips_available:
            # 仅在计算时将LPIPS移到GPU，用完移回CPU
            self.lpips.to(pred.device)
            # +++ 归一化输入到[-1,1] +++
            pred_lpips = 2 * pred - 1  # [0,1] → [-1,1]
            target_lpips = 2 * target - 1
            loss = self.lpips(pred_lpips, target_lpips).mean()
            return loss
        else:
            if self.vgg is None:
                # 延迟初始化并仅在需要时加载
                self.vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(pred.device)
                for param in self.vgg.parameters():
                    param.requires_grad = False
                self.slice1 = nn.Sequential(*list(self.vgg[:2])).to(pred.device)
                # self.slice2 = nn.Sequential(*list(self.vgg[2:7])).to(pred.device) # 示例：取2-6层

            # 标准化处理（与原逻辑一致）
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
            pred = (pred - mean) / std
            target = (target - mean) / std

            # 提取特征并计算损失（与原逻辑一致）
            features = []
            pred_feat = self.slice1(pred)
            target_feat = self.slice1(target)
            features.append((self.adaptive_pool(pred_feat), self.adaptive_pool(target_feat)))

            # pred_feat = self.slice2(pred_feat)
            # target_feat = self.slice2(target_feat)
            # features.append((self.adaptive_pool(pred_feat), self.adaptive_pool(target_feat)))

            loss = 0
            for (p, t) in features:
                loss += F.mse_loss(p, t)

            return loss


# 频率域损失
class FrequencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # 计算DCT变换后的差异
        pred_dct = torch.fft.fft2(pred, dim=(-2, -1))
        target_dct = torch.fft.fft2(target, dim=(-2, -1))

        # 计算幅度谱
        pred_mag = torch.abs(pred_dct)
        target_mag = torch.abs(target_dct)

        # 低频和高频分别计算损失
        h, w = pred.shape[-2], pred.shape[-1]
        low_freq_mask = torch.zeros((h, w), device=pred.device)
        center_h, center_w = h // 2, w // 2
        low_freq_range = min(h, w) // 4  # 低频区域大小
        low_freq_mask[center_h - low_freq_range:center_h + low_freq_range,
        center_w - low_freq_range:center_w + low_freq_range] = 1

        high_freq_mask = 1 - low_freq_mask

        low_freq_loss = F.l1_loss(pred_mag * low_freq_mask, target_mag * low_freq_mask)
        high_freq_loss = F.l1_loss(pred_mag * high_freq_mask, target_mag * high_freq_mask)

        # 调整权重，更注重高频细节（对PSNR和SSIM更有利）
        return low_freq_loss * 0.2 + high_freq_loss * 0.8


# 噪声感知损失
class NoiseAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, noise_residual):
        # 基础重建损失
        base_loss = F.l1_loss(pred, target)

        # 噪声一致性损失：预测图像与目标图像的噪声特性应该相似
        # 使用 avg_pool2d 的近似值，并添加 clamp 和 epsilon 防止除零和极端值
        pred_blur = F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1)
        pred_blur = torch.clamp(pred_blur, min=1e-4, max=1-1e-4) # 限制模糊后的值在合理范围内
        pred_noise = pred - pred_blur

        target_blur = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
        target_blur = torch.clamp(target_blur, min=1e-4, max=1-1e-4)
        target_noise = target - target_blur

        noise_loss = F.l1_loss(pred_noise, target_noise)

        # 噪声分布损失
        pred_noise_std = torch.std(pred_noise.view(pred_noise.shape[0], -1), dim=1)
        target_noise_std = torch.std(target_noise.view(target_noise.shape[0], -1), dim=1)
        # 防止 std 为 0
        pred_noise_std = torch.clamp(pred_noise_std, min=1e-6)
        target_noise_std = torch.clamp(target_noise_std, min=1e-6)
        std_loss = F.l1_loss(pred_noise_std, target_noise_std)

        return base_loss + 0.1 * noise_loss + 0.05 * std_loss


# +++ 修改：在Discriminator的CNN结构中应用谱归一化 +++
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, feat_channels=64):
        super().__init__()
        self.cnn = nn.Sequential(
            # 将每个Conv2d层用spectral_norm包装
            spectral_norm(nn.Conv2d(in_channels, feat_channels, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(feat_channels, feat_channels * 2, 4, stride=2, padding=1)),
            nn.InstanceNorm2d(feat_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(feat_channels * 2, feat_channels * 4, 4, stride=2, padding=1)),
            nn.InstanceNorm2d(feat_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(feat_channels * 4, feat_channels * 8, 4, stride=2, padding=1)),
            nn.InstanceNorm2d(feat_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(feat_channels * 8, 1, 4, stride=1, padding=1))
        )

    def forward(self, x):
        # 添加输入值范围检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("警告：判别器输入包含NaN或Inf值！")
            x = torch.clamp(x, -1.0, 1.0)  # 强制裁剪到合理范围

        # 添加梯度监控
        with torch.autocast('cuda', enabled=False):  # 禁用混合精度以确保数值稳定性
            x = self.cnn(x)

        # 更严格的输出限制
        return torch.clamp(x, -5.0, 5.0)  # 减少输出范围
class LossFunction(nn.Module):
    # 总损失函数：组合像素损失、平滑损失、纹理损失、亮度约束等
    def __init__(self):
        super(LossFunction, self).__init__()
        self._l2_loss = nn.MSELoss()  # 均方误差损失
        self._l1_loss = nn.L1Loss()  # 平均绝对误差损失
        self.smooth_loss = SmoothLoss()  # 光照平滑损失
        self.texture_difference = TextureDifference()  # 纹理差异损失
        self.local_mean = LocalMean(patch_size=5)  # 局部均值计算
        self.L_TV_loss = L_TV()  # 总变分(TV)损失
        self.perceptual_loss = ImprovedPerceptualLoss()  # 改进的感知损失
        self.ms_ssim_loss = MultiScaleSSIMLoss()  # 多尺度SSIM损失
        self.frequency_loss = FrequencyLoss()  # 频率域损失

        self.noise_aware_loss = NoiseAwareLoss()  # 噪声感知损失

        # 添加颜色一致性损失权重
        self.color_constancy_weight = 0.3
        self.histogram_match_weight = 0.25

        # 预生成直方图平滑用的高斯核
        self.hist_bins = 64
        self.hist_kernel_size = 5
        self.hist_bandwidth = 0.1
        kernel = torch.exp(-0.5 * (torch.linspace(-2, 2, self.hist_kernel_size) ** 2) / (self.hist_bandwidth ** 2))
        kernel = kernel / kernel.sum()
        self.register_buffer('hist_kernel', kernel.view(1, 1, -1))

        self.texture_preserve = TexturePreservationLoss()  # 纹理保留损失实例
        # 添加亮度监控参数
        self.brightness_threshold = 0.9  # 降低亮度阈值从0.92到0.9
        self.overexposure_weight = 0.1  # 增加过曝惩罚权重从0.3到0.5
        self.underexposure_threshold = 0.15
        # 添加详细的损失记录
        self.loss_components_detail = {}
        self.loss_components = {}
        # 动态权重参数 - 调整以提高PSNR和SSIM
        self.dynamic_weights = {
    'pixel_reconstruction': {'initial': 1.5, 'final': 0.8, 'transition_epoch': 2000},
    'perceptual': {'initial': 0.1, 'final': 0.8, 'transition_epoch': 2000},
    'texture_preserve': {'initial': 0.2, 'final': 0.5, 'transition_epoch': 2000},
    'color_constancy': {'initial': 0.05, 'final': 0.1, 'transition_epoch': 2000},
    'histogram_match': {'initial': 0.05, 'final': 0.2, 'transition_epoch': 2000},
    'ms_ssim': {'initial': 0.3, 'final': 1.0, 'transition_epoch': 2000},
    'frequency': {'initial': 0.1, 'final': 0.2, 'transition_epoch': 2000},
    'noise_aware': {'initial': 0.3, 'final': 0.5, 'transition_epoch': 2000}
}

        # 当前权重值
        self.current_weights = {key: config['initial'] for key, config in self.dynamic_weights.items()}

        # 添加损失记录字典
        self.loss_components = {}

    def ssim(self, x, y, window_size=11, size_average=True):
        """SSIM计算，与MultiScaleSSIMLoss中的实现一致"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = F.avg_pool2d(x, window_size, 1, 0)
        mu_y = F.avg_pool2d(y, window_size, 1, 0)

        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_x_mu_y = mu_x * mu_y

        sigma_x_sq = F.avg_pool2d(x * x, window_size, 1, 0) - mu_x_sq
        sigma_y_sq = F.avg_pool2d(y * y, window_size, 1, 0) - mu_y_sq
        sigma_xy = F.avg_pool2d(x * y, window_size, 1, 0) - mu_x_mu_y

        ssim_map = ((2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)) / (
                (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
        )

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    # 添加权重更新方法
    def update_weights(self, epoch):
        # 更精细的权重调度
        transition_epoch = 2000

        if epoch < 500:
            # 初期：注重基础重建
            self.current_weights = {
                'pixel_reconstruction': 1.5,
                'perceptual': 0.1,
                'texture_preserve': 0.2,
                'color_constancy': 0.05,
                'histogram_match': 0.05,
                'ms_ssim': 0.3,
                'frequency': 0.1,
                'noise_aware': 0.3
            }
        elif epoch < transition_epoch:
            # 过渡期：线性调整
            alpha = (epoch - 500) / (transition_epoch - 500)
            self.current_weights = {
                'pixel_reconstruction': 1.5 - 0.7 * alpha,
                'perceptual': 0.1 + 0.7 * alpha,
                'texture_preserve': 0.2 + 0.3 * alpha,
                'color_constancy': 0.05 + 0.05 * alpha,
                'histogram_match': 0.05 + 0.15 * alpha,
                'ms_ssim': 0.3 + 0.7 * alpha,
                'frequency': 0.1 + 0.1 * alpha,
                'noise_aware': 0.3 + 0.2 * alpha
            }
        else:
            # 后期：注重感知质量
            self.current_weights = {
                'pixel_reconstruction': 0.8,
                'perceptual': 0.8,
                'texture_preserve': 0.5,
                'color_constancy': 0.1,
                'histogram_match': 0.2,
                'ms_ssim': 1.0,
                'frequency': 0.2,
                'noise_aware': 0.5
            }

    def forward(self, input, target, epoch=0, **kwargs):

        # 数据范围检查
        assert torch.all(input >= -0.1) and torch.all(
            input <= 1.1), f"输入数据超出范围: {input.min().item():.4f} - {input.max().item():.4f}"
        assert torch.all(target >= -0.1) and torch.all(
            target <= 1.1), f"目标数据超出范围: {target.min().item():.4f} - {target.max().item():.4f}"
        self.smooth_factor = min(1.0, epoch / 1000)  # 逐渐增加平滑因子
        # 更新权重
        self.avg_brightness = 0
        self.overexposure_ratio = 0
        input = input.float()
        target = target.float()
        self.update_weights(epoch)
        eps = 1e-9

        # 重置详细记录
        self.loss_components_detail = {}

        # 检查所有输入是否有效
        for key, value in kwargs.items():
            if torch.is_tensor(value):
                kwargs[key] = value.float()  # 确保所有输入都是float32
                if torch.isnan(value).any() or torch.isinf(value).any():
                    print(f"输入 {key} 包含无效值，使用零替代")
                    kwargs[key] = torch.where(
                        torch.isnan(value) | torch.isinf(value),
                        torch.zeros_like(value),
                        value
                    )

        # 确保输入在合理范围内
        input = torch.clamp(input + eps, 0, 1)
        target = torch.clamp(target, 0, 1)  # 确保target也在[0,1]范围内
        self.update_weights(epoch)

        # 从 kwargs 中提取所需参数，使用get方法提供默认值
        L_pred1 = kwargs.get('L_pred1', torch.zeros_like(input))
        L_pred2 = kwargs.get('L_pred2', torch.zeros_like(input))
        L2 = kwargs.get('L2', torch.zeros_like(input))
        s2 = kwargs.get('s2', torch.zeros_like(input))
        s21 = kwargs.get('s21', torch.zeros_like(input))
        s22 = kwargs.get('s22', torch.zeros_like(input))
        H2 = kwargs.get('H2', torch.zeros_like(input))
        H11 = kwargs.get('H11', torch.zeros_like(input))
        H12 = kwargs.get('H12', torch.zeros_like(input))
        H13 = kwargs.get('H13', torch.zeros_like(input))
        s13 = kwargs.get('s13', torch.zeros_like(input))
        H14 = kwargs.get('H14', torch.zeros_like(input))
        s14 = kwargs.get('s14', torch.zeros_like(input))
        H3 = kwargs.get('H3', torch.zeros_like(input))
        s3 = kwargs.get('s3', torch.zeros_like(input))
        H3_pred = kwargs.get('H3_pred', torch.zeros_like(input))
        H4_pred = kwargs.get('H4_pred', torch.zeros_like(input))
        L_pred1_L_pred2_diff = kwargs.get('L_pred1_L_pred2_diff', torch.zeros_like(input))
        H3_denoised1_H3_denoised2_diff = kwargs.get('H3_denoised1_H3_denoised2_diff', torch.zeros_like(input))
        H2_blur = kwargs.get('H2_blur', torch.zeros_like(input))
        H3_blur = kwargs.get('H3_blur', torch.zeros_like(input))
        H3_denoised1 = kwargs.get('H3_denoised1', torch.zeros_like(input))
        H3_denoised2 = kwargs.get('H3_denoised2', torch.zeros_like(input))
        alpha_pred = kwargs.get('alpha_pred', torch.zeros(input.size(0), device=input.device))
        beta_pred = kwargs.get('beta_pred', torch.zeros(input.size(0), device=input.device))
        noise_residual = kwargs.get('noise_residual', torch.zeros_like(input))
        noise_prob = kwargs.get('noise_prob', torch.zeros((input.size(0), 3), device=input.device))

        input = input + eps  # 避免除以零

        # 1. 亮度增强约束与归一化约束
        # 标准 RGB 转灰度公式: R*0.299 + G*0.587 + B*0.114
        input_Y = L2.detach()[:, 0] * 0.299 + L2.detach()[:, 1] * 0.587 + L2.detach()[:, 2] * 0.114
        input_Y_mean = torch.mean(input_Y, dim=(1, 2))
        enhancement_factor = 0.5 / (input_Y_mean + eps)
        enhancement_factor = enhancement_factor.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        enhancement_factor = torch.clamp(enhancement_factor, 1, 10)
        adjustment_ratio = torch.pow(0.7, -enhancement_factor) / enhancement_factor
        adjustment_ratio = torch.clamp(adjustment_ratio, 0.1, 10)  # 添加钳位
        adjustment_ratio = adjustment_ratio.repeat(1, 3, 1, 1)

        normalized_low_light = L2.detach() / (s2 + eps)
        normalized_low_light = torch.clamp(normalized_low_light, eps, 1-eps)
        enhanced_brightness = torch.pow(L2.detach() * enhancement_factor, enhancement_factor)
        clamped_enhanced = torch.clamp(enhanced_brightness * adjustment_ratio, eps, 1)
        clamped_adjusted_low = torch.clamp(L2.detach() * enhancement_factor, eps, 1)

        loss = 0.0

        # 亮度整体约束损失（基于动态α与β预测）
        pix_loss, smooth_loss, total_ie_loss = ie_loss(s2, L2, alpha_pred, beta_pred)
        loss += total_ie_loss * 1
        self.loss_components['ie_loss'] = total_ie_loss.item()
        # 添加详细记录
        self.loss_components_detail['ie_loss'] = {
            'value': total_ie_loss.item(),
            'components': {
                'pix_loss': pix_loss.item(),
                'smooth_loss': smooth_loss.item()
            }
        }

        # 归一化低光层与增强亮度目标的约束
        norm_loss = self._l2_loss(normalized_low_light, clamped_adjusted_low) * 100
        loss += norm_loss
        self.loss_components['norm_loss'] = norm_loss.item()
        self.loss_components_detail['norm_loss'] = {
            'value': norm_loss.item(),
            'components': None
        }

        # 2. 多尺度去噪一致性损失
        L11_small, L12_small = pair_downsampler(input)
        loss1 = self._l2_loss(L11_small, L_pred2) * 10
        loss2 = self._l2_loss(L12_small, L_pred1) * 10
        loss += loss1 + loss2
        self.loss_components['downsample_loss1'] = loss1.item()
        self.loss_components['downsample_loss2'] = loss2.item()
        self.loss_components_detail['downsample_loss1'] = {
            'value': loss1.item(),
            'components': None
        }
        self.loss_components_detail['downsample_loss2'] = {
            'value': loss2.item(),
            'components': None
        }

        denoised1, denoised2 = pair_downsampler(L2)
        loss3 = self._l2_loss(L_pred1, denoised1) * 50
        loss4 = self._l2_loss(L_pred2, denoised2) * 50
        loss += loss3 + loss4
        self.loss_components['denoise_loss1'] = loss3.item()
        self.loss_components['denoise_loss2'] = loss4.item()
        self.loss_components_detail['denoise_loss1'] = {
            'value': loss3.item(),
            'components': None
        }
        self.loss_components_detail['denoise_loss2'] = {
            'value': loss4.item(),
            'components': None
        }

        # 3. 残差尺寸对齐一致性损失
        target_H3 = torch.cat([H12.detach(), s22.detach()], dim=1)
        if H3_pred.shape[2:] != target_H3.shape[2:]:
            H3_pred = F.interpolate(H3_pred, size=target_H3.shape[2:], mode='bilinear', align_corners=True)
        align_loss1 = self._l2_loss(H3_pred, target_H3) * 50
        loss += align_loss1
        self.loss_components['align_loss1'] = align_loss1.item()
        self.loss_components_detail['align_loss1'] = {
            'value': align_loss1.item(),
            'components': None
        }

        target_H4 = torch.cat([H11.detach(), s21.detach()], dim=1)
        if H4_pred.shape[2:] != target_H4.shape[2:]:
            H4_pred = F.interpolate(H4_pred, size=target_H4.shape[2:], mode='bilinear', align_corners=True)
        align_loss2 = self._l2_loss(H4_pred, target_H4) * 50
        loss += align_loss2
        self.loss_components['align_loss2'] = align_loss2.item()
        self.loss_components_detail['align_loss2'] = {
            'value': align_loss2.item(),
            'components': None
        }

        # 4. 颜色一致性损失（模糊后保证颜色分布一致）
        color_loss = self._l2_loss(H2_blur.detach(), H3_blur) * 100
        loss += color_loss
        self.loss_components['color_loss'] = color_loss.item()
        self.loss_components_detail['color_loss'] = {
            'value': color_loss.item(),
            'components': None
        }

        # 5. 光照一致性损失
        illumination_loss = self._l2_loss(s2.detach(), s3) * 10
        loss += illumination_loss
        self.loss_components['illumination_loss'] = illumination_loss.item()
        self.loss_components_detail['illumination_loss'] = {
            'value': illumination_loss.item(),
            'components': None
        }

        # 6. 内容一致性损失（局部均值约束）
        local_mean1 = self.local_mean(H3_denoised1)
        local_mean2 = self.local_mean(H3_denoised2)
        weighted_diff1 = (
                                 1 - H3_denoised1_H3_denoised2_diff) * local_mean1 + H3_denoised1 * H3_denoised1_H3_denoised2_diff
        weighted_diff2 = (
                                 1 - H3_denoised1_H3_denoised2_diff) * local_mean2 + H3_denoised2 * H3_denoised1_H3_denoised2_diff
        content_loss1 = self._l2_loss(H3_denoised1, weighted_diff1) * 50
        content_loss2 = self._l2_loss(H3_denoised2, weighted_diff2) * 50
        loss += content_loss1 + content_loss2
        self.loss_components['content_loss1'] = content_loss1.item()
        self.loss_components['content_loss2'] = content_loss2.item()
        self.loss_components_detail['content_loss1'] = {
            'value': content_loss1.item(),
            'components': None
        }
        self.loss_components_detail['content_loss2'] = {
            'value': content_loss2.item(),
            'components': None
        }

        # 7. 噪声方差约束损失
        noise_std = calculate_local_variance(H3 - H2)
        H2_var = calculate_local_variance(H2)
        noise_var_loss = self._l2_loss(H2_var, noise_std) * 50
        loss += noise_var_loss
        self.loss_components['noise_var_loss'] = noise_var_loss.item()
        self.loss_components_detail['noise_var_loss'] = {
            'value': noise_var_loss.item(),
            'components': None
        }


        # 8. 基础像素重建损失（使用动态权重）
        pred_img = H3  # 最终的去噪输出图像

        # 添加范围检查和处理
        pred_img = torch.clamp(pred_img, 0, 1)
        target = torch.clamp(target, 0, 1)

        # 使用更稳定的MSE计算
        rd_loss = F.mse_loss(pred_img, target)
        # 添加SSIM损失作为辅助
        ssim_loss_val = 1 - self.ssim(pred_img, target)
        # 组合损失
        reconstruction_loss = rd_loss + 0.3 * ssim_loss_val

        loss += self.current_weights['pixel_reconstruction'] * reconstruction_loss
        self.loss_components['pixel_reconstruction'] = reconstruction_loss.item()
        self.loss_components_detail['pixel_reconstruction'] = {
            'value': reconstruction_loss.item(),
            'components': {
                'mse_loss': rd_loss.item(),
                'ssim_loss': ssim_loss_val.item()
            }
        }

        # 9. 感知损失（使用动态权重）
        perceptual_loss_val = self.perceptual_loss(pred_img, target)
        # 应用平滑
        perceptual_loss_val = perceptual_loss_val * self.smooth_factor + \
                              perceptual_loss_val.detach() * (1 - self.smooth_factor)
        loss += self.current_weights['perceptual'] * perceptual_loss_val
        self.loss_components['perceptual'] = perceptual_loss_val.item()
        self.loss_components_detail['perceptual'] = {
            'value': perceptual_loss_val.item(),
            'components': None
        }

        # 10. 纹理保留损失（使用动态权重）
        texture_loss = self.texture_preserve(input, H3)
        loss += self.current_weights['texture_preserve'] * texture_loss
        self.loss_components['texture_preserve'] = texture_loss.item()
        self.loss_components_detail['texture_preserve'] = {
            'value': texture_loss.item(),
            'components': None
        }

        # 11. 颜色一致性损失（使用动态权重）
        H2_color = kwargs.get('H2_color', None)
        if H2_color is not None:
            color_loss = self.color_constancy_loss(H2_color)
            loss += self.current_weights['color_constancy'] * color_loss
            self.loss_components['color_constancy'] = color_loss.item()
            self.loss_components_detail['color_constancy'] = {
                'value': color_loss.item(),
                'components': None
            }

        # 12. 直方图匹配损失（使用动态权重）
        H3_for_hist = kwargs.get('H3', None)
        if H3_for_hist is not None:
            hist_loss = self.histogram_match_loss(H3_for_hist, target)
            loss += self.current_weights['histogram_match'] * hist_loss
            self.loss_components['histogram_match'] = hist_loss.item()
            self.loss_components_detail['histogram_match'] = {
                'value': hist_loss.item(),
                'components': None
            }

        # 13. 多尺度SSIM损失（增加权重以提高SSIM）
        ms_ssim_loss_val = self.ms_ssim_loss(pred_img, target)
        loss += self.current_weights['ms_ssim'] * ms_ssim_loss_val
        self.loss_components['ms_ssim'] = ms_ssim_loss_val.item()
        self.loss_components_detail['ms_ssim'] = {
            'value': ms_ssim_loss_val.item(),
            'components': None
        }

        # 14. 频率域损失（调整权重分配）
        freq_loss_val = self.frequency_loss(pred_img, target)
        loss += self.current_weights['frequency'] * freq_loss_val
        self.loss_components['frequency'] = freq_loss_val.item()
        self.loss_components_detail['frequency'] = {
            'value': freq_loss_val.item(),
            'components': None
        }

        # 16. 噪声感知损失（新增）
        noise_aware_loss_val = self.noise_aware_loss(pred_img, target, noise_residual)
        loss += self.current_weights['noise_aware'] * noise_aware_loss_val
        self.loss_components['noise_aware'] = noise_aware_loss_val.item()
        self.loss_components_detail['noise_aware'] = {
            'value': noise_aware_loss_val.item(),
            'components': None
        }

        # 添加噪声分类损失（如果提供了真实噪声标签）
        noise_type_label = kwargs.get('noise_type_label', None)
        if noise_type_label is not None:
            noise_cls_loss = F.cross_entropy(noise_prob, noise_type_label)
            loss += 0.1 * noise_cls_loss
            self.loss_components['noise_classification'] = noise_cls_loss.item()
            self.loss_components_detail['noise_classification'] = {
                'value': noise_cls_loss.item(),
                'components': None
            }

        # 5. 亮度约束与过曝控制（关键修改）
        # 计算当前输出图像的亮度
        brightness = 0.299 * pred_img[:, 0] + 0.587 * pred_img[:, 1] + 0.114 * pred_img[:, 2]
        avg_brightness = torch.mean(brightness)
        # 记录到self，用于日志打印
        self.avg_brightness = avg_brightness

        # a. 欠曝光惩罚：如果平均亮度低于阈值，则施加惩罚
        if avg_brightness < self.underexposure_threshold:
            underexposure_loss = (self.underexposure_threshold - avg_brightness) * 2.0
            loss += underexposure_loss
            self.loss_components['underexposure_loss'] = underexposure_loss.item()
            self.loss_components_detail['underexposure_loss'] = {
                'value': underexposure_loss.item(),
                'components': None
            }

        # b. 过曝光惩罚：惩罚过亮的像素
        overexposure_mask = (brightness > self.brightness_threshold).float()
        self.overexposure_ratio = torch.mean(overexposure_mask)  # 记录过曝比例
        overexposure_loss = torch.mean(overexposure_mask * (brightness - self.brightness_threshold) ** 2)
        loss += self.overexposure_weight * overexposure_loss
        self.loss_components['overexposure_loss'] = overexposure_loss.item()
        self.loss_components_detail['overexposure_loss'] = {
            'value': overexposure_loss.item(),
            'components': None
        }

        # 记录亮度统计信息（用于日志）
        self.avg_brightness = torch.mean(pred_img)

        if not torch.is_tensor(loss):
            loss = torch.tensor(loss, device=input.device, dtype=torch.float32, requires_grad=True)

        # 记录总损失
        self.loss_components['total_loss'] = loss.item()
        self.loss_components_detail['total_loss'] = {
            'value': loss.item(),
            'components': None
        }

        return loss

    def get_loss_components(self):
        """获取损失组件的字典"""
        return self.loss_components

    def get_detailed_loss_components(self):
        """获取详细的损失组件信息"""
        return self.loss_components_detail

    # 优化后的颜色恒常性损失
    def color_constancy_loss(self, x):
        """颜色恒常性损失：减少色偏（确保批次内每个样本独立计算）"""
        # x shape: (batch, 3, h, w)
        mean_r = torch.mean(x[:, 0, :, :], dim=(1, 2))  # shape: (batch,)
        mean_g = torch.mean(x[:, 1, :, :], dim=(1, 2))
        mean_b = torch.mean(x[:, 2, :, :], dim=(1, 2))

        diff_rg = torch.square(mean_r - mean_g)
        diff_rb = torch.square(mean_r - mean_b)
        diff_gb = torch.square(mean_g - mean_b)

        return torch.mean(torch.sqrt(diff_rg + diff_rb + diff_gb + 1e-8))

    def adaptive_brightness_constraint(self, pred_img):
        # 计算亮度（RGB转灰度的加权和）
        brightness = 0.299 * pred_img[:, 0] + 0.587 * pred_img[:, 1] + 0.114 * pred_img[:, 2]
        # 计算每个样本的平均亮度（按空间维度求均值）
        avg_brightness = torch.mean(brightness, dim=(1, 2))

        # 更温和的亮度调整：以目标亮度0.4为基准
        target_brightness = 0.45
        # 计算调整比例，避免除零
        brightness_ratio = target_brightness / (avg_brightness + 1e-6)
        brightness_ratio = torch.where(avg_brightness < target_brightness,
                                       torch.clamp(brightness_ratio, 1.0, 1.5),  # 欠曝最多提1.5倍
                                       torch.clamp(brightness_ratio, 0.8, 1.0))  # 过曝只降不升

        # 应用亮度调整（广播到图像维度）
        adjusted_img = pred_img * brightness_ratio.view(-1, 1, 1, 1)
        # 4. 新增欠曝惩罚（对亮度<0.2的像素额外惩罚）
        underexposed = (brightness < 0.2).float()
        underexpose_penalty = torch.mean(underexposed * (0.2 - brightness) ** 2)
        self.underexpose_penalty = underexpose_penalty  # 用于后续损失叠加
        # 确保像素值在有效范围[0,1]内
        return torch.clamp(adjusted_img, 0, 1)

    # 修复后的直方图匹配损失
    def histogram_match_loss(self, pred, target, bins=None):
        bins = self.hist_bins if bins is None else bins
        loss = 0.0
        pred_clamped = torch.clamp(pred, 0.0, 1.0)
        target_clamped = torch.clamp(target, 0.0, 1.0)

        # 确保直方图核在与输入相同的设备上
        hist_kernel = self.hist_kernel.to(pred.device)

        for c in range(3):
            # 计算归一化直方图 - 确保在正确设备上
            pred_hist = torch.histc(pred_clamped[:, c].flatten(), bins=bins, min=0.0, max=1.0)
            pred_hist = pred_hist.to(pred.device)  # 确保在相同设备
            pred_hist = pred_hist / (pred_hist.sum() + 1e-8)

            target_hist = torch.histc(target_clamped[:, c].flatten(), bins=bins, min=0.0, max=1.0)
            target_hist = target_hist.to(pred.device)  # 确保在相同设备
            target_hist = target_hist / (target_hist.sum() + 1e-8)

            # 高斯平滑（设备一致）
            pred_smoothed = F.conv1d(
                pred_hist.view(1, 1, -1),
                hist_kernel,
                padding=(self.hist_kernel_size - 1) // 2
            ).squeeze()

            target_smoothed = F.conv1d(
                target_hist.view(1, 1, -1),
                hist_kernel,
                padding=(self.hist_kernel_size - 1) // 2
            ).squeeze()

            loss += F.l1_loss(pred_smoothed, target_smoothed)

        return loss / 3


def ie_loss(s, i, alpha_pred, beta_pred):
    # 使用预测的动态参数，而非固定计算

    gamma = 0.7
    eps = 1e-6
    # 像素强度调整损失 - 使用预测的alpha和beta
    # 将形状为 [B] 的 alpha_pred 和 beta_pred 扩展为 [B, 1, 1, 1] 以匹配图像张量 s 和 i 的形状 [B, C, H, W]
    alpha_expanded = alpha_pred[:, None, None, None]  # 等同于 .unsqueeze(1).unsqueeze(2).unsqueeze(3)
    beta_expanded = beta_pred[:, None, None, None]

    # 钳位输入值
    i_clamped = torch.clamp(i, eps, 1 - eps)
    alpha_i = torch.clamp(alpha_expanded * i_clamped, min=eps)
    # 计算像素损失
    pix_loss = F.mse_loss(s, beta_expanded * (alpha_i + eps) ** gamma)

    # 平滑损失
    grad_h = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
    grad_w = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])
    smooth_loss = grad_h.mean() + grad_w.mean()

    total_loss = pix_loss + 0.01 * smooth_loss  # 总损失
    return pix_loss, smooth_loss, total_loss  # 返回子分量和总损失


class TextureDifference(nn.Module):
    # 计算两张图像的纹理差异
    def __init__(self, patch_size=5, constant_C=1e-5, threshold=0.975):
        super(TextureDifference, self).__init__()
        self.patch_size = patch_size
        self.constant_C = constant_C
        self.threshold = threshold

    def forward(self, image1, image2):
        eps = 1e-8
        # 转灰度
        image1 = self.rgb_to_gray(image1)
        image2 = self.rgb_to_gray(image2)
        # 计算局部标准差（纹理变化程度）
        stddev1 = self.local_stddev(image1)
        stddev2 = self.local_stddev(image2)
        numerator = 2 * stddev1 * stddev2
        denominator = stddev1 ** 2 + stddev2 ** 2 + self.constant_C + eps
        diff = numerator / denominator  # 范围[0,1]
        # 超过阈值的视为纹理一致（记为1），否则为0
        binary_diff = torch.where(diff > self.threshold,
                                  torch.tensor(1.0, device=diff.device),
                                  torch.tensor(0.0, device=diff.device))
        return binary_diff

    # 修复缩进：确保这两个方法在类内部
    def local_stddev(self, image):
        padding = self.patch_size // 2
        image = F.pad(image, (padding, padding, padding, padding), mode='reflect')
        patches = image.unfold(2, self.patch_size, 1).unfold(3, self.patch_size, 1)
        mean = patches.mean(dim=(4, 5), keepdim=True)
        squared_diff = (patches - mean) ** 2
        local_var = squared_diff.mean(dim=(4, 5))
        local_std = torch.sqrt(local_var + 1e-9)
        return local_std

    def rgb_to_gray(self, image):
        gray_image = 0.144 * image[:, 0] + 0.587 * image[:, 1] + 0.299 * image[:, 2]
        return gray_image.unsqueeze(1)


class TexturePreservationLoss(nn.Module):
    def __init__(self, edge_weight=0.8):
        super().__init__()
        self.edge_weight = edge_weight
        # Sobel算子用于边缘检测
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_x.weight.data = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y.weight.data = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        # 冻结参数
        self.sobel_x.weight.requires_grad = True
        self.sobel_y.weight.requires_grad = True

    def forward(self, input, output):
        # 转为灰度图
        input_gray = 0.299 * input[:, 0] + 0.587 * input[:, 1] + 0.114 * input[:, 2]
        output_gray = 0.299 * output[:, 0] + 0.587 * output[:, 1] + 0.114 * output[:, 2]

        # 计算梯度幅度
        input_grad_x = self.sobel_x(input_gray.unsqueeze(1))
        input_grad_y = self.sobel_y(input_gray.unsqueeze(1))
        input_grad_mag = torch.sqrt(input_grad_x ** 2 + input_grad_y ** 2 + 1e-6)

        output_grad_x = self.sobel_x(output_gray.unsqueeze(1))
        output_grad_y = self.sobel_y(output_gray.unsqueeze(1))
        output_grad_mag = torch.sqrt(output_grad_x ** 2 + output_grad_y ** 2 + 1e-6)

        # 梯度相似性损失
        grad_loss = F.l1_loss(output_grad_mag, input_grad_mag)

        # 结构相似性损失（SSIM）
        ssim_loss = 1 - self.ssim(output, input)

        return self.edge_weight * grad_loss + (1 - self.edge_weight) * ssim_loss

    def ssim(self, x, y, window_size=11, size_average=True):
        # 简化SSIM实现
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = F.avg_pool2d(x, window_size, 1, 0)
        mu_y = F.avg_pool2d(y, window_size, 1, 0)

        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_x_mu_y = mu_x * mu_y

        sigma_x_sq = F.avg_pool2d(x * x, window_size, 1, 0) - mu_x_sq
        sigma_y_sq = F.avg_pool2d(y * y, window_size, 1, 0) - mu_y_sq
        sigma_xy = F.avg_pool2d(x * y, window_size, 1, 0) - mu_x_mu_y

        ssim_map = ((2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)) / (
                (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


class L_TV(nn.Module):
    # 总变分损失，用于保持图像平滑
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size(0)
        h_x = x.size(2)
        w_x = x.size(3)
        count_h = (h_x - 1) * w_x
        count_w = h_x * (w_x - 1)
        h_tv = ((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]) ** 2).sum()
        w_tv = ((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]) ** 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class Blur(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.nc = nc
        kernel_tensor = gauss_kernel(kernlen=21, nsig=3, channels=self.nc)
        weight = kernel_tensor.float()
        self.register_buffer('weight', weight)

    def forward(self, x):
        if x.size(1) != self.nc:
            raise RuntimeError(f"输入通道数[{x.size(1)}]与预设[{self.nc}]不匹配")
        return F.conv2d(x, self.weight, stride=1, padding=10, groups=self.nc)


class SmoothLoss(nn.Module):
    # 平滑损失：约束光照图的空间平滑性（基于输入图像颜色相似性）
    def __init__(self):
        super(SmoothLoss, self).__init__()
        self.sigma = 10

    def rgb2yCbCr(self, input_im):
        im_flat = input_im.contiguous().view(-1, 3).float()
        device = input_im.device
        mat = torch.tensor([[0.257, -0.148, 0.439],
                            [0.564, -0.291, -0.368],
                            [0.098, 0.439, -0.071]], device=device)
        bias = torch.tensor([16 / 255., 128 / 255., 128 / 255.], device=device)
        temp = im_flat @ mat + bias
        out = temp.view(input_im.shape[0], 3, input_im.shape[2], input_im.shape[3])
        return out

    def forward(self, input, output):
        # input: 原始图像; output: 光照图s2
        self.output = output
        self.input = self.rgb2yCbCr(input)
        sigma_color = -1.0 / (2 * self.sigma * self.sigma)
        # 计算各方向的颜色相似性权重
        w1 = torch.exp(
            torch.sum((self.input[:, :, 1:, :] - self.input[:, :, :-1, :]) ** 2, dim=1, keepdim=True) * sigma_color)
        w2 = torch.exp(
            torch.sum((self.input[:, :, :-1, :] - self.input[:, :, 1:, :]) ** 2, dim=1, keepdim=True) * sigma_color)
        w3 = torch.exp(
            torch.sum((self.input[:, :, :, 1:] - self.input[:, :, :, :-1]) ** 2, dim=1, keepdim=True) * sigma_color)
        w4 = torch.exp(
            torch.sum((self.input[:, :, :, :-1] - self.input[:, :, :, 1:]) ** 2, dim=1, keepdim=True) * sigma_color)
        w5 = torch.exp(
            torch.sum((self.input[:, :, :-1, :-1] - self.input[:, :, 1:, 1:]) ** 2, dim=1, keepdim=True) * sigma_color)
        w6 = torch.exp(
            torch.sum((self.input[:, :, 1:, 1:] - self.input[:, :, :-1, :-1]) ** 2, dim=1, keepdim=True) * sigma_color)
        w7 = torch.exp(
            torch.sum((self.input[:, :, 1:, :-1] - self.input[:, :, :-1, 1:]) ** 2, dim=1, keepdim=True) * sigma_color)
        w8 = torch.exp(
            torch.sum((self.input[:, :, :-1, 1:] - self.input[:, :, 1:, :-1]) ** 2, dim=1, keepdim=True) * sigma_color)
        w9 = torch.exp(
            torch.sum((self.input[:, :, 2:, :] - self.input[:, :, :-2, :]) ** 2, dim=1, keepdim=True) * sigma_color)
        w10 = torch.exp(
            torch.sum((self.input[:, :, :-2, :] - self.input[:, :, 2:, :]) ** 2, dim=1, keepdim=True) * sigma_color)
        w11 = torch.exp(
            torch.sum((self.input[:, :, :, 2:] - self.input[:, :, :, :-2]) ** 2, dim=1, keepdim=True) * sigma_color)
        w12 = torch.exp(
            torch.sum((self.input[:, :, :, :-2] - self.input[:, :, :, 2:]) ** 2, dim=1, keepdim=True) * sigma_color)
        w13 = torch.exp(
            torch.sum((self.input[:, :, :-2, :-1] - self.input[:, :, 2:, 1:]) ** 2, dim=1, keepdim=True) * sigma_color)
        w14 = torch.exp(
            torch.sum((self.input[:, :, 2:, 1:] - self.input[:, :, :-2, :-1]) ** 2, dim=1, keepdim=True) * sigma_color)
        w15 = torch.exp(
            torch.sum((self.input[:, :, 2:, :-1] - self.input[:, :, :-2, 1:]) ** 2, dim=1, keepdim=True) * sigma_color)
        w16 = torch.exp(
            torch.sum((self.input[:, :, :-2, 1:] - self.input[:, :, 2:, :-1]) ** 2, dim=1, keepdim=True) * sigma_color)
        w17 = torch.exp(torch.sum((self.input[:, :, :-1, :-2] - self.input[:, :, 1:, 2:], 2) ** 2, dim=1,
                                  keepdim=True) * sigma_color)
        w18 = torch.exp(
            torch.sum((self.input[:, :, 1:, 2:] - self.input[:, :, :-1, :-2]) ** 2, dim=1, keepdim=True) * sigma_color)
        w19 = torch.exp(
            torch.sum((self.input[:, :, 1:, :-2] - self.input[:, :, :-1, 2:]) ** 2, dim=1, keepdim=True) * sigma_color)
        w20 = torch.exp(
            torch.sum((self.input[:, :, :-1, 2:] - self.input[:, :, 1:, :-2]) ** 2, dim=1, keepdim=True) * sigma_color)
        w21 = torch.exp(
            torch.sum((self.input[:, :, :-2, :-2] - self.input[:, :, 2:, 2:]) ** 2, dim=1, keepdim=True) * sigma_color)

        w22 = torch.exp(
            torch.sum((self.input[:, :, 2:, 2:] - self.input[:, :, :-2, :-2]) ** 2, dim=1, keepdim=True) * sigma_color)
        w23 = torch.exp(
            torch.sum((self.input[:, :, 2:, :-2] - self.input[:, :, :-2, 2:]) ** 2, dim=1, keepdim=True) * sigma_color)
        w24 = torch.exp(
            torch.sum((self.input[:, :, :-2, 2:] - self.input[:, :, 2:, :-2]) ** 2, dim=1, keepdim=True) * sigma_color)
        # 计算光照图在各方向的加权差异
        pixel_grad1 = w1 * torch.norm(self.output[:, :, 1:, :] - self.output[:, :, :-1, :], p=1, dim=1, keepdim=True)
        pixel_grad2 = w2 * torch.norm(self.output[:, :, :-1, :] - self.output[:, :, 1:, :], p=1, dim=1, keepdim=True)
        pixel_grad3 = w3 * torch.norm(self.output[:, :, :, 1:] - self.output[:, :, :, :-1], p=1, dim=1, keepdim=True)
        pixel_grad4 = w4 * torch.norm(self.output[:, :, :, :-1] - self.output[:, :, :, 1:], p=1, dim=1, keepdim=True)
        pixel_grad5 = w5 * torch.norm(self.output[:, :, :-1, :-1] - self.output[:, :, 1:, 1:], p=1, dim=1, keepdim=True)
        pixel_grad6 = w6 * torch.norm(self.output[:, :, 1:, 1:] - self.output[:, :, :-1, :-1], p=1, dim=1, keepdim=True)
        pixel_grad7 = w7 * torch.norm(self.output[:, :, 1:, :-1] - self.output[:, :, :-1, 1:], p=1, dim=1, keepdim=True)
        pixel_grad8 = w8 * torch.norm(self.output[:, :, :-1, 1:] - self.output[:, :, 1:, :-1], p=1, dim=1, keepdim=True)
        pixel_grad9 = w9 * torch.norm(self.output[:, :, 2:, :] - self.output[:, :, :-2, :], p=1, dim=1, keepdim=True)
        pixel_grad10 = w10 * torch.norm(self.output[:, :, :-2, :] - self.output[:, :, 2:, :], p=1, dim=1, keepdim=True)
        pixel_grad11 = w11 * torch.norm(self.output[:, :, :, 2:] - self.output[:, :, :, :-2], p=1, dim=1, keepdim=True)
        pixel_grad12 = w12 * torch.norm(self.output[:, :, :, :-2] - self.output[:, :, :, 2:], p=1, dim=1, keepdim=True)
        pixel_grad13 = w13 * torch.norm(self.output[:, :, :-2, :-1] - self.output[:, :, 2:, 1:], p=1, dim=1,
                                        keepdim=True)
        pixel_grad14 = w14 * torch.norm(self.output[:, :, 2:, 1:] - self.output[:, :, :-2, :-1], p=1, dim=1,
                                        keepdim=True)
        pixel_grad15 = w15 * torch.norm(self.output[:, :, 2:, :-1] - self.output[:, :, :-2, 1:], p=1, dim=1,
                                        keepdim=True)
        pixel_grad16 = w16 * torch.norm(self.output[:, :, :-2, 1:] - self.output[:, :, 2:, :-1], p=1, dim=1,
                                        keepdim=True)
        pixel_grad17 = w17 * torch.norm(self.output[:, :, :-1, :-2] - self.output[:, :, 1:, 2:], p=1, dim=1,
                                        keepdim=True)
        pixel_grad18 = w18 * torch.norm(self.output[:, :, 1:, 2:] - self.output[:, :, :-1, :-2], p=1, dim=1,
                                        keepdim=True)
        pixel_grad19 = w19 * torch.norm(self.output[:, :, 1:, :-2] - self.output[:, :, :-1, 2:], p=1, dim=1,
                                        keepdim=True)
        pixel_grad20 = w20 * torch.norm(self.output[:, :, :-1, 2:] - self.output[:, :, 1:, :-2], p=1, dim=1,
                                        keepdim=True)
        pixel_grad21 = w21 * torch.norm(self.output[:, :, :-2, :-2] - self.output[:, :, 2:, 2:], p=1, dim=1,
                                        keepdim=True)
        pixel_grad22 = w22 * torch.norm(self.output[:, :, 2:, 2:] - self.output[:, :, :-2, :-2], p=1, dim=1,
                                        keepdim=True)
        pixel_grad23 = w23 * torch.norm(self.output[:, :, 2:, :-2] - self.output[:, :, :-2, 2:], p=1, dim=1,
                                        keepdim=True)
        pixel_grad24 = w24 * torch.norm(self.output[:, :, :-2, 2:] - self.output[:, :, 2:, :-2], p=1, dim=1,
                                        keepdim=True)
        # 平滑损失：所有方向差异的平均和
        reg_term = (pixel_grad1.mean() + pixel_grad2.mean() + pixel_grad3.mean() + pixel_grad4.mean() +
                    pixel_grad5.mean() + pixel_grad6.mean() + pixel_grad7.mean() + pixel_grad8.mean() +
                    pixel_grad9.mean() + pixel_grad10.mean() + pixel_grad11.mean() + pixel_grad12.mean() +
                    pixel_grad13.mean() + pixel_grad14.mean() + pixel_grad15.mean() + pixel_grad16.mean() +
                    pixel_grad17.mean() + pixel_grad18.mean() + pixel_grad19.mean() + pixel_grad20.mean() +
                    pixel_grad21.mean() + pixel_grad22.mean() + pixel_grad23.mean() + pixel_grad24.mean())
        return reg_term