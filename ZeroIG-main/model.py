import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

from utils import blur, pair_downsampler  # 导入工具函数：模糊处理、下采样
from torch.utils.checkpoint import checkpoint
from loss import LossFunction, TextureDifference, Discriminator  # 导入损失函数相关类
from utils import gauss_kernel


# 噪声分类器：识别噪声类型（高斯/泊松/椒盐）
class NoiseClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        # 添加额外的全连接层
        self.extra_fc1 = nn.Linear(32, 16)
        self.extra_fc2 = nn.Linear(16, 3)

    def forward(self, noise_residual):
        # 使用完整CNN序列
        x = self.cnn(noise_residual)
        x = F.relu(self.extra_fc1(x))
        x = self.extra_fc2(x)
        prob = F.softmax(x, dim=1)
        # 添加clone以防止inplace修改
        return prob.clone()


# 改进的IE-Net（Enhancer类），引入注意力模块
class Enhancer(nn.Module):
    def __init__(self, layers=8, channels=64):
        super().__init__()
        self.init_conv = nn.Conv2d(5, channels, 3, padding=1)

        # 添加空间-通道注意力模块
        self.attention_modules = nn.ModuleList()
        for _ in range(layers):
            self.attention_modules.append(AttentionModule(channels))

        # 添加亮度约束模块
        self.brightness_control = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 32, 1),  # 64→32（中间维度按比例增加）
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1),  # 增加一层卷积增强特征
                nn.ReLU(),
                self.attention_modules[i]  # 使用注意力模块
            ))
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels, 3, 3, padding=1),
            nn.Sigmoid()  # 增加 Sigmoid 激活，将输出压缩到 [0, 1]
        )

    def forward(self, input, alpha_pred, beta_pred):
        B, C, H, W = input.shape
        alpha_map = alpha_pred.view(B, 1, 1, 1).expand(B, 1, H, W)
        beta_map = beta_pred.view(B, 1, 1, 1).expand(B, 1, H, W)

        conditioned_input = torch.cat([input, alpha_map, beta_map], dim=1)
        fea = self.init_conv(conditioned_input)

        # 亮度控制
        brightness_factor = self.brightness_control(fea)
        brightness_factor = torch.clamp(brightness_factor, 0.8, 3.5)  # 限制亮度调整范围

        # 应用带注意力的块
        for i, block in enumerate(self.blocks):
            fea = fea + block(fea)
            # 在特定层后应用注意力
            if i % 2 == 1:  # 每隔一层应用额外注意力
                fea = self.attention_modules[i](fea)

        fea = self.final_conv(fea)
        brightness_factor = torch.clamp(brightness_factor, 1.5, 6.0)  # 原0.8-3.5
        fea = fea * brightness_factor
        fea = torch.clamp(fea, 0, 1.0)   # 允许轻微过曝（1.2），避免过暗

        return fea


# 动态参数预测器：根据亮度直方图和噪声水平预测α和β
class DynamicParamPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入：亮度直方图（假设100 bins）+ 噪声水平（1个值）
        self.fc = nn.Sequential(
            nn.Linear(101, 64),  # 100 bins + 1噪声水平
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # 输出α_pred和β_pred
        )

    def forward(self, hist, noise_level):
        # 确保输入在同一设备上
        assert hist.device == noise_level.device, "Hist and noise_level must be on the same device"
        # 输入处理：直方图展平 + 噪声水平拼接
        hist_flat = hist.view(hist.shape[0], -1)  # [B, 100]
        input_feat = torch.cat([hist_flat, noise_level.unsqueeze(1)], dim=1)  # [B, 101]
        params = self.fc(input_feat)  # [B, 2]

        # +++ 新增：约束alpha和beta为非负 +++
        alpha_pred_raw, beta_pred_raw = params.split(1, dim=1)
        alpha_pred = torch.relu(alpha_pred_raw).squeeze(1)  # 确保α≥0
        beta_pred = torch.relu(beta_pred_raw).squeeze(1)  # 确保β≥0（避免负光照）
        return alpha_pred, beta_pred





# 多尺度空间-通道注意力模块
class AttentionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 通道注意力分支
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
            nn.Sigmoid()
        )
        # 空间注意力分支
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        channel_weight = self.channel_att(x)  # [B, C, 1, 1]
        x = x * channel_weight
        # 空间注意力
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_feat = torch.cat([max_pool, avg_pool], dim=1)
        spatial_weight = self.spatial_att(spatial_feat)  # [B, 1, H, W]
        x = x * spatial_weight
        return x


class Denoise_1(nn.Module):
    # 第一级去噪模块（轻量级卷积网络）
    def __init__(self, chan_embed=48):
        super(Denoise_1, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(3, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, 3, 1)

    def forward(self, x):
        x = checkpoint(self.conv1, x, use_reentrant=False)
        x = self.act(x)
        x = checkpoint(self.conv2, x, use_reentrant=False)
        x = self.act(x)
        x = checkpoint(self.conv3, x, use_reentrant=False)
        return x


# +++ 添加：Transformer编码器（全局特征建模） +++
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, max_seq_len=16384):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        # 添加可学习的位置编码
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))

    def forward(self, x):
        # x shape:
        seq_len = x.size(1)
        # 添加位置编码
        x = x + self.pos_encoding[:, :seq_len, :]
        attn_output, _ = self.attention(x, x, x)
        return attn_output


# +++ 修改：改进的RD-Net（混合架构，替代原Denoise_2） +++
class Denoise2(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        # CNN局部特征提取
        self.texture_extractor = nn.Sequential(
            nn.Conv2d(3, channels // 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels // 2, channels // 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.texture_proj = nn.Conv2d(channels // 2, channels, 1)  # 1x1卷积调整通道数
        self.cnn = nn.Sequential(
            nn.Conv2d(6, channels, 3, padding=1),  # 输入：反射图(3)+光照图(3)
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        self.down_ratio = 4  # 从8减少到4，保留更多细节
        self.transformer_norm = nn.LayerNorm(channels)
        self.transformer = TransformerEncoder(embed_dim=channels, max_seq_len=16384)
        self.fusion = nn.Conv2d(channels * 2, channels, 1)
        self.attn = AttentionModule(channels)
        self.final_conv = nn.Conv2d(channels, 6, 1)
        self.noise_classifier = NoiseClassifier()
        self.gauss_conv = nn.Conv2d(channels, 6, 1)
        self.poisson_conv = nn.Conv2d(channels, 6, 1)
        self.salt_conv = nn.Conv2d(channels, 6, 1)

    def _gaussian_blur(self, x, kernel_size=3, sigma=1.0):
        channels = x.shape[1]
        kernel = gauss_kernel(kernel_size, sigma, channels, device=x.device)
        padding = kernel_size // 2
        return F.conv2d(x, kernel, padding=padding, groups=channels)

    def _resize_if_needed(self, tensor, target):
        if tensor.shape[2:] != target.shape[2:]:
            return F.interpolate(tensor, size=target.shape[2:], mode='bilinear', align_corners=False)
        return tensor

    def forward(self, r, s, noise_residual):
        noise_prob = self.noise_classifier(noise_residual)
        gauss_prob, poisson_prob, salt_prob = noise_prob[:, 0], noise_prob[:, 1], noise_prob[:, 2]

        texture_feat = self.texture_extractor(r)
        texture_feat = self.texture_proj(texture_feat)
        texture_feat = self._resize_if_needed(texture_feat, r)

        x = torch.cat([r, s], dim=1)
        cnn_feat = self.cnn(x)
        cnn_feat = cnn_feat + 0.3 * texture_feat

        cnn_feat_down = F.avg_pool2d(cnn_feat, kernel_size=self.down_ratio, stride=self.down_ratio)

        B, C, H_down, W_down = cnn_feat_down.shape
        seq_len = H_down * W_down
        max_seq_len = self.transformer.pos_encoding.size(1)

        if seq_len > max_seq_len:
            additional_down_ratio = int(np.ceil(np.sqrt(seq_len / max_seq_len)))
            cnn_feat_down = F.avg_pool2d(cnn_feat_down, kernel_size=additional_down_ratio, stride=additional_down_ratio)

        transformer_input = cnn_feat_down.flatten(2).permute(0, 2, 1)
        transformer_feat = self.transformer(transformer_input)

        B, _, C = transformer_feat.shape
        H_new, W_new = H_down, W_down

        transformer_feat = transformer_feat.permute(0, 2, 1).reshape(B, C, H_new, W_new)

        transformer_feat = F.interpolate(transformer_feat, size=cnn_feat.shape[2:], mode='bilinear',
                                         align_corners=False)

        B, C, H, W = transformer_feat.shape
        transformer_feat_norm_input = transformer_feat.reshape(B, C, H * W).permute(0, 2, 1)
        transformer_feat_norm = self.transformer_norm(transformer_feat_norm_input)
        transformer_feat = transformer_feat_norm.permute(0, 2, 1).reshape(B, C, H, W)

        fused = self.fusion(torch.cat([cnn_feat, transformer_feat], dim=1))
        fused = self.attn(fused)

        gauss_out = self.gauss_conv(fused)
        poisson_feat = torch.sqrt(F.relu(fused) + 1e-6)
        poisson_out = self.poisson_conv(poisson_feat)
        salt_mid = self._gaussian_blur(fused, kernel_size=3, sigma=1.0)
        salt_out = self.salt_conv(salt_mid)

        gauss_out = self._resize_if_needed(gauss_out, r)
        poisson_out = self._resize_if_needed(poisson_out, r)
        salt_out = self._resize_if_needed(salt_out, r)

        gauss_weight = gauss_prob.view(-1, 1, 1, 1)
        poisson_weight = poisson_prob.view(-1, 1, 1, 1)
        salt_weight = salt_prob.view(-1, 1, 1, 1)

        total_weight = gauss_weight + poisson_weight + salt_weight + 1e-6
        gauss_weight = gauss_weight / total_weight
        poisson_weight = poisson_weight / total_weight
        salt_weight = salt_weight / total_weight

        combined = (gauss_out * gauss_weight + poisson_out * poisson_weight + salt_out * salt_weight)
        return torch.clamp(combined, 0, 1.0)
# 可学习亮度校正模块（替代启发式亮度调整）
class LearnableBrightnessCorrection(nn.Module):
    def __init__(self):
        super().__init__()
        # 可学习的全局缩放因子和偏移量（初始值设为1.0和0.0，即不改变原始亮度）
        self.scale = nn.Parameter(torch.tensor(1.2))
        self.shift = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        # 仿射变换调整亮度，确保输出在[0,1]范围内
        corrected = x * self.scale + self.shift
        return torch.clamp(corrected, 0.0, 1.0)


# +++ 增强版颜色校正模块 +++
class EnhancedColorCorrection(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, 3, 1)
        )
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3, 16, 1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 1),
            nn.Sigmoid()
        )

    def forward(self, input, enhanced):
        # 确保输入在有效范围内
        input = torch.clamp(input, 0, 1)
        enhanced = torch.clamp(enhanced, 1e-4, 1)  # 避免除零

        # 连接特征
        concat_feat = torch.cat([input, enhanced], dim=1)

        # 应用校正
        correction = self.conv(concat_feat)
        attn_map = self.attention(input)
        corrected = enhanced + correction * attn_map

        # 确保输出在有效范围内
        return torch.clamp(corrected, 0, 1)


# 相机响应函数校正模块

class CRFCorrection(nn.Module):
    """相机响应函数校正模块"""
    def __init__(self, init_gamma=0.45, learnable=True):
        super().__init__()
        if learnable:
            self.gamma = nn.Parameter(torch.tensor(init_gamma))
        else:
            self.register_buffer('gamma', torch.tensor(init_gamma))

        # 可学习的色调映射曲线
        # 使用 LeakyReLU 替代 ReLU，防止梯度消失
        self.curve = nn.Sequential(
            nn.Linear(1, 8),
            nn.LeakyReLU(0.01, inplace=True),  # 替换 nn.ReLU()
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        # 初始化曲线网络的权重
        self._initialize_weights()

    def _initialize_weights(self):
        """
        专门为这个小MLP设计初始化。
        使用Xavier初始化对于Linear层搭配LeakyReLU是较好的选择。
        """
        for m in self.curve.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # 将偏置初始化为一个小的正值，增加初始阶段的活性
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        # Gamma校正
        x_gamma = torch.pow(x, self.gamma)

        # 可学习的曲线调整（逐像素）
        B, C, H, W = x.shape
        x_flat = x.reshape(B * C * H * W, 1)
        x_curve = self.curve(x_flat).reshape(B, C, H, W)

        # 混合输出
        return 0.7 * x_gamma + 0.3 * x_curve




class Network(nn.Module):
    # 主网络（训练时使用）
    def __init__(self, debug=False):
        super(Network, self).__init__()
        self.debug = debug  # 调试模式标志
        self.enhance = Enhancer(layers=8, channels=64)  # 增强模块
        self.denoise_1 = Denoise_1(chan_embed=16)  # 第一级去噪
        self.denoise_2 = Denoise2(channels=64)  # 第二级去噪
        self.param_predictor = DynamicParamPredictor()  # 动态参数预测器
        self.noise_classifier = NoiseClassifier()  # 噪声分类器
        # 判别器及损失函数
        self.discriminator = Discriminator()
        self._criterion = LossFunction()
        # 其它辅助模块
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.TextureDifference = TextureDifference()
        self.color_correct = EnhancedColorCorrection()  # 使用增强版颜色校正模块
        # 添加CRF校正模块
        self.crf_correction = CRFCorrection(learnable=True)
        self.brightness_correction = LearnableBrightnessCorrection()

        # 添加中间层监控
        self.intermediate_outputs = {}
        self._register_hooks()
        # 检查参数初始化
        for name, param in self.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"参数 {name} 包含NaN或Inf值，重新初始化")
                nn.init.xavier_uniform_(param.data)

    def _register_hooks(self):
        """注册前向钩子来监控中间层输出"""

        def get_activation(name):
            def hook(model, input, output):
                self.intermediate_outputs[name] = {
                    'min': output.min().item(),
                    'max': output.max().item(),
                    'mean': output.mean().item(),
                    'std': output.std().item()
                }

            return hook

        # 监控关键层
        layers_to_monitor = {
            'enhance.init_conv': self.enhance.init_conv,
            'enhance.blocks.0': self.enhance.blocks[0],
            'denoise_1.conv1': self.denoise_1.conv1,
            'denoise_1.conv2': self.denoise_1.conv2,
            'denoise_1.conv3': self.denoise_1.conv3,
            'denoise_2.cnn': self.denoise_2.cnn,
            'denoise_2.transformer': self.denoise_2.transformer,
            'param_predictor.fc.0': self.param_predictor.fc[0],
            'param_predictor.fc.2': self.param_predictor.fc[2],
            'param_predictor.fc.4': self.param_predictor.fc[4],
            'noise_classifier.cnn': self.noise_classifier.cnn,
            'color_correct.conv.0': self.color_correct.conv[0],
            'color_correct.conv.3': self.color_correct.conv[3],
            'color_correct.conv.6': self.color_correct.conv[6],
            'crf_correction.curve.0': self.crf_correction.curve[0],
            'crf_correction.curve.2': self.crf_correction.curve[2],
        }

        for name, layer in layers_to_monitor.items():
            layer.register_forward_hook(get_activation(name))

    def get_intermediate_outputs(self):
        """获取中间层输出信息"""
        return self.intermediate_outputs

    def enhance_weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)
            if m.bias is not None:  # 修正为 bias
                m.bias.data.zero_()
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1, 0.02)

    def denoise_weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:  # 修正为 bias
                m.bias.data.zero_()
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1, 0.02)

    def _compute_brightness_histogram(self, x, bins=100):
        """计算输入图像的亮度直方图（转为灰度后计算）"""
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]  # [B, H, W]
        hist_list = []
        for i in range(gray.shape[0]):
            hist = torch.histc(gray[i], bins=bins, min=0, max=1)
            hist = hist / (gray.shape[1] * gray.shape[2])
            hist_list.append(hist)
        return torch.stack(hist_list, dim=0)  # [B, 100]

    def _estimate_noise_level(self, x):
        """简单估计噪声水平（输入图像与模糊版本的差异）"""
        x_blur = blur(x)
        noise = x - x_blur
        return torch.mean(torch.abs(noise), dim=[1, 2, 3])  # [B]

    def _debug_print(self, name, tensor):
        """调试打印函数"""
        if self.debug:
            print(f"{name}: shape={tensor.shape}, min={tensor.min().item():.4f}, "
                  f"max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f}, "
                  f"has_nan={torch.isnan(tensor).any().item()}, "
                  f"has_inf={torch.isinf(tensor).any().item()}")

    def forward(self, input):
        # 清空中间层输出记录
        self.intermediate_outputs = {}

        outputs = {}  # 初始化字典
        eps = 1e-4
        input = input + eps
        input.requires_grad_(True)

        # 调试输入
        self._debug_print("Input", input)

        # 计算亮度直方图和噪声水平
        brightness_hist = self._compute_brightness_histogram(input)
        noise_level = self._estimate_noise_level(input)
        alpha_pred, beta = self.param_predictor(brightness_hist, noise_level)

        # 调试参数预测器
        if self.debug:
            print(f"Alpha_pred: {alpha_pred.mean().item():.4f}, Beta: {beta.mean().item():.4f}")

        # 第一级去噪
        noise_residual = checkpoint(self.denoise_1, input, use_reentrant=False)
        self._debug_print("Noise_residual", noise_residual)

        noise_prob = self.noise_classifier(noise_residual)
        if self.debug:
            print(f"Noise_prob: {noise_prob.mean(dim=0)}")

        # 下采样输入图像（构建多尺度）
        L11, L12 = pair_downsampler(input)
        L_pred1 = L11 - checkpoint(self.denoise_1, L11, use_reentrant=False)
        L_pred2 = L12 - checkpoint(self.denoise_1, L12, use_reentrant=False)
        L2 = input - noise_residual
        L2 = torch.clamp(L2, eps, 1)
        self._debug_print("L2", L2)

        # 增强模块生成光照图
        s2 = checkpoint(self.enhance, L2, alpha_pred, beta, use_reentrant=False)  # 传入两个参数
        # CRF校正
        s2 = self.crf_correction(s2)
        s2 = torch.clamp(s2, 0.01, 1)
        self._debug_print("s2", s2)

        s21, s22 = pair_downsampler(s2)

        # +++ 关键修改：确保H2反射图被钳位在[0,1]范围内 +++
        s2_clamped = torch.clamp(s2, min=0.01)  # 使用一个更大、更安全的最小值
        H2 = input / (s2_clamped + 1e-6)
        H2 = torch.clamp(H2, 0, 1.0)  # 同时将输出也钳位到有效范围内
        self._debug_print("H2", H2)

        # 对增强后的反射图H2进行颜色校正（输入原图input和增强图H2）
        H2_color = self.color_correct(input, H2)
        H2_color = torch.clamp(H2_color, 0, 1.0)
        outputs['H2_color'] = H2_color  # 将校正结果加入输出
        self._debug_print("H2_color", H2_color)

        # 多尺度增强的反射图
        s21_clamped = torch.clamp(s21, min=0.01)
        H11 = L11 / s21_clamped
        H11 = torch.clamp(H11, 0, 1.0)

        s22_clamped = torch.clamp(s22, min=0.01)
        H12 = L12 / s22_clamped
        H12 = torch.clamp(H12, 0, 1.0)

        # 第二级去噪（多尺度输入）
        H3_pred = self.denoise_2(H11, s21, noise_residual)
        H3_pred = torch.clamp(H3_pred, eps, 1)
        self._debug_print("H3_pred", H3_pred)

        H13 = H3_pred[:, :3, :, :]
        s13 = H3_pred[:, 3:, :, :]
        H4_pred = self.denoise_2(H12, s22, noise_residual)
        H4_pred = torch.clamp(H4_pred, eps, 1)
        self._debug_print("H4_pred", H4_pred)

        H14 = H4_pred[:, :3, :, :]
        s14 = H4_pred[:, 3:, :, :]
        H5_pred = self.denoise_2(H2, s2, noise_residual)
        H5_pred = torch.clamp(H5_pred, eps, 1)
        self._debug_print("H5_pred", H5_pred)

        H3 = H5_pred[:, :3, :, :]
        enhanced_final = H3 * s2  # 这是最终的增强结果
        enhanced_final = torch.clamp(enhanced_final, 1e-4, 1.0)
        s3 = H5_pred[:, 3:, :, :]
        # 应用可学习亮度校正（新增代码）
        enhanced_final = self.brightness_correction(enhanced_final)
        # 更新H3为校正后的值（如果H3作为最终输出）
        H3 = enhanced_final

        self._debug_print("H3 (final output)", H3)  # 确认H3来自Denoise2

        # 纹理差异计算（用于损失）
        L_pred1_L_pred2_diff = self.TextureDifference(L_pred1, L_pred2)
        H3_denoised1, H3_denoised2 = pair_downsampler(H3)
        H3_denoised1_H3_denoised2_diff = self.TextureDifference(H3_denoised1, H3_denoised2)

        # 计算模糊版本（用于颜色一致性损失）
        H1 = L2 / (s2 + 1e-8)
        H1 = torch.clamp(H1, 0, 1)
        H2_blur = blur(H1)
        H3_blur = blur(H3)

        # 明确主输出和辅助输出
        return {
            'enhanced': enhanced_final,  # 作为主输出
            'illumination': s2,  # 光照图作为辅助输出
            'denoised': H3,  # 去噪结果
            'L_pred1': L_pred1,
            'L_pred2': L_pred2,
            'L2': L2,
            's2': s2,
            's21': s21,
            's22': s22,
            'H2': H2,
            'H2_color': H2_color,  # 颜色校正后的增强图
            'H11': H11,
            'H12': H12,
            'H13': H13,
            's13': s13,
            'H14': H14,
            's14': s14,
            'H3_denoised1': H3_denoised1,
            'H3_denoised2': H3_denoised2,
            'H3': H3,  # 最终输出，来自Denoise2
            's3': s3,
            'H3_pred': H3_pred,
            'H4_pred': H4_pred,
            'L_pred1_L_pred2_diff': L_pred1_L_pred2_diff,
            'H3_denoised1_H3_denoised2_diff': H3_denoised1_H3_denoised2_diff,
            'H2_blur': H2_blur,
            'H3_blur': H3_blur,
            'alpha_pred': alpha_pred,
            'beta_pred': beta,
            'noise_prob': noise_prob,
            'noise_residual': noise_residual
        }

    def _loss(self, input, target, epoch=0, **outputs):
        # 计算总损失（组合多种损失项）
        return self._criterion(input, target, epoch=epoch, **outputs)


class Finetunemodel(nn.Module):
    # 微调模型（测试时使用）
    def __init__(self, weights, debug=False):
        super(Finetunemodel, self).__init__()
        self.debug = debug  # 调试模式标志
        self.enhance = Enhancer(layers=8, channels=64)
        self.denoise_1 = Denoise_1(chan_embed=32)
        self.denoise_2 = Denoise2(channels=64)
        self.param_predictor = DynamicParamPredictor()
        self.noise_classifier = NoiseClassifier()
        # +++ 添加增强版颜色校正模块 +++
        self.color_correct = EnhancedColorCorrection()
        # 添加CRF校正模块
        self.crf_correction = CRFCorrection(learnable=True)
        # 加载预训练权重
        base_weights = torch.load(weights, map_location='cpu')
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in base_weights.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        # 添加TTA标志
        self.use_tta = False  # 默认启用测试时增强

    def _debug_print(self, name, tensor):
        """调试打印函数"""
        if self.debug:
            print(f"{name}: shape={tensor.shape}, min={tensor.min().item():.4f}, "
                  f"max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f}, "
                  f"has_nan={torch.isnan(tensor).any().item()}, "
                  f"has_inf={torch.isinf(tensor).any().item()}")

    def _compute_brightness_histogram(self, x, bins=100):
        """计算输入图像的亮度直方图（转为灰度后计算）"""
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]  # [B, H, W]
        hist_list = []
        for i in range(gray.shape[0]):
            # 先将数据移动到CPU计算直方图，然后再移回原设备
            hist = torch.histc(gray[i].cpu(), bins=bins, min=0, max=1)
            hist = hist.to(x.device)  # 移回GPU
            hist = hist / (gray.shape[1] * gray.shape[2])
            hist_list.append(hist)
        return torch.stack(hist_list, dim=0)  # [B, 100]

    def _estimate_noise_level(self, x):
        """估计噪声水平（输入与模糊图之差）"""
        x_blur = blur(x)
        noise = x - x_blur
        return torch.mean(torch.abs(noise), dim=[1, 2, 3])

    def _forward_impl(self, input):
        """单次前向传播实现"""
        eps = 1e-4
        input = input + eps
        # 调试输入
        self._debug_print("Input", input)

        # 使用梯度检查点包装计算密集型操作
        def compute_features(x):
            brightness_hist = self._compute_brightness_histogram(x)
            noise_level = self._estimate_noise_level(x)
            return self.param_predictor(brightness_hist, noise_level)

        alpha_pred, beta = checkpoint(compute_features, input)

        # 调试参数预测器
        if self.debug:
            print(f"Alpha_pred: {alpha_pred.mean().item():.4f}, Beta: {beta.mean().item():.4f}")

        # 第一级去噪与噪声分类
        noise_residual = self.denoise_1(input)
        self._debug_print("Noise_residual", noise_residual)

        noise_prob = self.noise_classifier(noise_residual)
        if self.debug:
            print(f"Noise_prob: {noise_prob.mean(dim=0)}")

        # 计算去噪后图像
        L2 = input - noise_residual
        L2 = torch.clamp(L2, eps, 1)
        self._debug_print("L2", L2)

        # 增强模块生成光照图
        s2 = checkpoint(self.enhance, L2, alpha_pred, beta)
        # CRF校正
        s2 = self.crf_correction(s2)
        s2 = torch.clamp(s2, eps, 1)
        self._debug_print("s2", s2)

        # 计算增强后的反射图
        H2 = input / (s2 + 1e-8)
        H2 = torch.clamp(H2, 0, 1.2) # 修改：从[0,2]改为[0,1]
        self._debug_print("H2", H2)

        # 第二级去噪（RD-Net）
        H5_pred = checkpoint(self.denoise_2, H2, s2, noise_residual)
        H5_pred = torch.clamp(H5_pred, eps, 1)
        self._debug_print("H5_pred", H5_pred)

        H3 = H5_pred[:, :3, :, :]  # 最终去噪结果
        self._debug_print("H3 (final output)", H3)  # 确认H3来自Denoise2

        # +++ 添加后处理：通道调整 +++
        # 蓝色通道增强（索引2为蓝色通道）
        H3[:, 2] = H3[:, 2] * 1.05
        # 红色通道减弱（索引0为红色通道）
        H3[:, 0] = H3[:, 0] * 0.97
        # 确保数值仍在[0, 1]范围内
        H3 = torch.clamp(H3, eps, 1)
        self._debug_print("H3 (after color adjustment)", H3)

        # 返回与训练阶段对应的输出
        return {
            'enhanced': H3,  # 主输出
            'illumination': s2,  # 辅助输出
            'H2': H2,  # 增强图
            'H3': H3,  # 处理后的去噪图
            'alpha_pred': alpha_pred,
            'beta_pred': beta,
            'noise_prob': noise_prob,
            'noise_residual': noise_residual
        }

    def forward(self, input):
        if not self.use_tta:
            # 不使用TTA，直接返回单次前向传播结果
            return self._forward_impl(input)

        # 测试时增强（TTA） - 多尺度融合
        # 原始尺度
        output1 = self._forward_impl(input)

        # 水平翻转
        output2 = self._forward_impl(torch.flip(input, [3]))
        output2['H2'] = torch.flip(output2['H2'], [3])
        output2['H3'] = torch.flip(output2['H3'], [3])
        output2['enhanced'] = torch.flip(output2['enhanced'], [3])

        # 垂直翻转
        output3 = self._forward_impl(torch.flip(input, [2]))
        output3['H2'] = torch.flip(output3['H2'], [2])
        output3['H3'] = torch.flip(output3['H3'], [2])
        output3['enhanced'] = torch.flip(output3['enhanced'], [2])

        # 多尺度平均
        final_output = {}
        for key in output1.keys():
            if isinstance(output1[key], torch.Tensor) and output1[key].dim() == 4:
                # 对图像输出进行平均
                final_output[key] = (output1[key] + output2[key] + output3[key]) / 3
            else:
                # 对其他输出保持原始值
                final_output[key] = output1[key]

        return final_output