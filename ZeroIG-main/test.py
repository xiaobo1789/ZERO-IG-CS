import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import argparse

import logging
import lpips as lpips_lib
import pyiqa
from PIL import Image
from torch.autograd import Variable
from model import Finetunemodel  # 导入微调模型类
from multi_read_data import DataLoader  # 导入自定义数据加载器
from thop import profile  # 用于计算模型FLOPs
from ultralytics import YOLO
import torchvision.transforms as T
import cv2

# 设置根目录路径，确保可以正确导入项目模块
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(root_dir)

# 解析命令行参数
parser = argparse.ArgumentParser("ZERO-IG")
parser.add_argument('--data_path_test_low', type=str, default='./data/LOL-V1/lol_dataset/eval15/cs/low',
                    help='测试低光图像路径')
parser.add_argument('--data_path_test_target', type=str, default='./data/LOL-V1/lol_dataset/eval15/cs/high',
                    help='（可选）测试目标图像路径，用于计算指标')
parser.add_argument('--save', type=str, default='./results/', help='结果保存目录')
parser.add_argument('--model_test', type=str, default='./EXP/Train-20250729-002842/model_epochs/weights_800.pt',
                    help='预训练模型权重路径')
parser.add_argument('--gpu', type=int, default=0, help='使用的GPU设备ID')
parser.add_argument('--seed', type=int, default=2, help='随机种子（保证结果可复现）')
parser.add_argument('--yolo_size', type=int, default=640, help='YOLO输入尺寸')
parser.add_argument('--tta', action='store_true', help='启用测试时增强')
args = parser.parse_args()

# 创建结果保存目录
save_path = args.save
os.makedirs(save_path, exist_ok=True)

# 配置日志输出到控制台和文件
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
metric_log = logging.FileHandler(os.path.join(save_path, 'log.txt'))
metric_log.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(metric_log)
logging.info("test file name = %s", os.path.split(__file__)[1])

# 加载测试数据集
TestDataset = DataLoader(img_dir=args.data_path_test_low,
                         target_dir=(args.data_path_test_target if args.data_path_test_target else None),
                         task='test')
test_queue = torch.utils.data.DataLoader(TestDataset, batch_size=1, pin_memory=True, num_workers=0, shuffle=False)


def save_images(tensor):
    # 将模型输出的张量转为可保存图像格式（支持单张）
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    im = np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8')
    return im


def calculate_model_parameters(model):
    # 计算模型参数总量
    return sum(p.numel() for p in model.parameters())


def calculate_model_flops(model, input_tensor):
    # 计算模型FLOPs（浮点运算次数）
    flops, _ = profile(model, inputs=(input_tensor,))
    return flops / 1e9  # 转换为GFLOPs


def adaptive_brightness_control(image, max_brightness=0.92, min_avg_brightness=0.35):
    brightness = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
    avg_brightness = torch.mean(brightness)

    # 更保守的亮度调整
    if avg_brightness < min_avg_brightness:
        scale = min_avg_brightness / (avg_brightness + 1e-6)
        scale = torch.clamp(scale, 1.0, 2.0)  # 限制最大提升2倍
        image = image * scale

    # 更精细的过曝处理
    overexposed = (brightness > max_brightness).float()
    overexposed_ratio = overexposed.mean()

    if overexposed_ratio > 0.08:
        # 使用平滑的过曝修复
        correction_mask = torch.clamp((brightness - max_brightness) / (1 - max_brightness), 0, 1)
        correction_strength = 0.1 + 0.4 * correction_mask  # 动态调整修复强度
        image = image * (1 - correction_strength.unsqueeze(1))

    return torch.clamp(image, 0, 1)


def estimate_noise_level(image):
    """估计图像噪声水平"""
    if image.dim() == 3:
        image = image.unsqueeze(0)

    # 转换为灰度图
    gray = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]

    # 计算局部方差
    local_var = torch.var(gray.unfold(2, 5, 1).unfold(3, 5, 1), dim=(-2, -1))

    # 噪声水平估计为局部方差的平均值
    noise_level = torch.mean(torch.sqrt(local_var + 1e-6))

    return noise_level.item()


def adaptive_denoise(enhanced_image, noise_level=0.1):
    """
    自适应后处理降噪
    noise_level: 0-1之间，控制降噪强度
    """
    # 转换为numpy格式
    if isinstance(enhanced_image, torch.Tensor):
        enhanced_image = enhanced_image.cpu().numpy()
        if enhanced_image.shape[0] == 3:
            enhanced_image = np.transpose(enhanced_image, (1, 2, 0))

    # 根据噪声水平选择降噪参数
    h = 3 + int(15 * noise_level)  # h值从3到18
    template_window_size = 7
    search_window_size = 21

    # 将 RGB 转换为 BGR 以供 OpenCV 处理
    enhanced_image_bgr = cv2.cvtColor((enhanced_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # 应用非局部均值降噪
    denoised_bgr = cv2.fastNlMeansDenoisingColored(
        enhanced_image_bgr,
        None,
        h, h, template_window_size, search_window_size
    )

    # 将 BGR 结果转换回 RGB
    denoised = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)

    return denoised.astype(np.float32) / 255.0


def evaluate_detection_mAP(enhanced_images, target_images, model, orig_size=(600, 400)):
    """使用 YOLO 计算增强图像和目标图像上的 mAP"""
    # 转换张量为YOLO可接受的输入格式 (0-255范围的RGB图像)
    transform = T.Compose([
        T.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        T.ToPILImage()
    ])

    # 处理增强图像
    enhanced_img = transform(enhanced_images.squeeze(0).cpu())
    # 处理目标图像
    target_img = transform(target_images.squeeze(0).cpu())

    # 保存原始尺寸
    enhanced_orig_size = enhanced_img.size
    target_orig_size = target_img.size

    # 执行检测
    enhanced_results = model(enhanced_img, verbose=False)
    target_results = model(target_img, verbose=False)

    # 计算mAP@0.5
    enhanced_map = enhanced_results[0].boxes.map50 if enhanced_results[0].boxes is not None else 0.0
    target_map = target_results[0].boxes.map50 if target_results[0].boxes is not None else 0.0

    return enhanced_map, target_map

def calculate_metrics(enhanced, target, device, psnr_metric, ssim_metric, lpips_model, niqe_metric, yolo_model, noise_level):
    """
    统一计算所有评估指标
    Args:
        enhanced: 增强后的图像张量 (已归一化到[0,1])
        target: 目标图像张量 (若存在，已归一化到[0,1])
        device: 计算设备
        psnr_metric/ssim_metric/lpips_model/niqe_metric: 指标计算模型
        yolo_model: YOLO检测模型
        noise_level: 噪声水平估计值
    Returns:
        包含所有指标的字典
    """
    metrics = {
        'psnr': None,
        'ssim': None,
        'lpips': None,
        'niqe': None,
        'noise_level': noise_level,
        'enhance_map': None,
        'target_map': None
    }

    # 计算NIQE（无参考指标，始终计算）
    metrics['niqe'] = niqe_metric(enhanced).item()

    # 若存在目标图像，计算全参考指标
    if target is not None:
        # 确保数据在相同设备
        enhanced = enhanced.to(device)
        target = target.to(device)

        # 计算PSNR、SSIM、LPIPS
        metrics['psnr'] = psnr_metric(enhanced, target).item()
        metrics['ssim'] = ssim_metric(enhanced, target).item()
        metrics['lpips'] = lpips_model(enhanced, target).mean().item()

        # 计算目标检测mAP
        enhance_map, target_map = evaluate_detection_mAP(enhanced, target, yolo_model)
        metrics['enhance_map'] = enhance_map
        metrics['target_map'] = target_map

    return metrics
def resize_for_yolo(image_pil, target_size=640):
    """
    调整图像尺寸以适应YOLO输入，保持宽高比并进行填充
    """
    # 计算缩放比例
    orig_width, orig_height = image_pil.size
    scale = min(target_size / orig_width, target_size / orig_height)

    # 计算新尺寸
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)

    # 调整图像大小
    resized = image_pil.resize((new_width, new_height), Image.BILINEAR)

    # 创建新图像并进行填充
    new_image = Image.new('RGB', (target_size, target_size), (114, 114, 114))
    new_image.paste(resized, ((target_size - new_width) // 2, (target_size - new_height) // 2))

    return new_image, scale, (target_size - new_width) // 2, (target_size - new_height) // 2


def main():
    if not torch.cuda.is_available():
        print('无可用GPU设备，测试终止。')
        sys.exit(1)
    # 设置所用设备和随机种子
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 初始化指标模型
    psnr_metric = pyiqa.create_metric('psnr', device=device)
    ssim_metric = pyiqa.create_metric('ssim', device=device)
    niqe_metric = pyiqa.create_metric('niqe', device=device)
    lpips_model = lpips_lib.LPIPS(net='alex').to(device)

    # 创建指标日志文件
    metric_log_path = os.path.join(save_path, 'metrics_log.txt')
    with open(metric_log_path, 'w') as f:
        f.write("Image Name, PSNR, SSIM, LPIPS, NIQE, Noise_Level, mAP(enhance), mAP(target)\n")

    # 加载预训练模型权重
    model = Finetunemodel(args.model_test)
    model = model.to(device)
    model.eval()
    model.use_tta = args.tta  # 根据参数启用TTA

    # 计算模型参数量并输出
    total_params = calculate_model_parameters(model)
    logging.info("总参数量: %f M", total_params / 1e6)
    # 冻结模型参数
    for p in model.parameters():
        p.requires_grad = False

    # 加载YOLO模型（只需加载一次）
    yolo_model = YOLO('yolov5s.pt').to(device)

    # YOLO模型预热
    logging.info("预热YOLO模型...")
    dummy_input = torch.randn(1, 3, args.yolo_size, args.yolo_size).to(device)
    _ = yolo_model(dummy_input)

    # 无梯度计算的推理
    with torch.no_grad():
        for _, batch in enumerate(test_queue):
            # 根据是否有目标图像，解析 batch
            if args.data_path_test_target:
                input_tensor, target_tensor, img_name = batch
                target_tensor = target_tensor.to(device)
            else:
                input_tensor, img_name = batch
                target_tensor = None
            input_tensor = input_tensor.to(device)
            input_name = os.path.splitext(os.path.basename(img_name[0]))[0]

            # 执行模型推理
            result = model(input_tensor)
            enhance_tensor = result['H2']  # 增强图像张量
            output_tensor = result['H3']  # 去噪图像张量

            # 应用自适应亮度控制后处理
            output_tensor = adaptive_brightness_control(output_tensor)

            # 估计噪声水平
            noise_level = estimate_noise_level(output_tensor)

            # 应用自适应降噪
            if noise_level > 0.05:  # 仅在噪声水平较高时应用降噪
                output_tensor_denoised = adaptive_denoise(output_tensor, noise_level)
                # 转换为Tensor
                if isinstance(output_tensor_denoised, np.ndarray):
                    output_tensor_denoised = torch.from_numpy(
                        np.transpose(output_tensor_denoised, (2, 0, 1))
                    ).unsqueeze(0).to(device)
                output_tensor = output_tensor_denoised

            # 保存输出图像
            enhance_img = save_images(enhance_tensor)
            output_img = save_images(output_tensor)
            os.makedirs(os.path.join(save_path, 'result'), exist_ok=True)
            Image.fromarray(output_img).save(os.path.join(save_path, 'result', f'{input_name}_denoise.png'), 'PNG')
            Image.fromarray(enhance_img).save(os.path.join(save_path, 'result', f'{input_name}_enhance.png'), 'PNG')

            # 确保输出张量在0-1范围内
            output_tensor_norm = torch.clamp(output_tensor, 0, 1)
            enhance_tensor_norm = torch.clamp(enhance_tensor, 0, 1)

            # 确保输出张量在0-1范围内
            output_tensor_norm = torch.clamp(output_tensor, 0, 1)
            enhance_tensor_norm = torch.clamp(enhance_tensor, 0, 1)

            # 计算指标（调用封装函数）
            metrics = calculate_metrics(
                enhanced=output_tensor_norm,  # 最终输出的增强图像（去噪后）
                target=target_tensor,  # 目标图像（可能为None）
                device=device,
                psnr_metric=psnr_metric,
                ssim_metric=ssim_metric,
                lpips_model=lpips_model,
                niqe_metric=niqe_metric,
                yolo_model=yolo_model,
                noise_level=noise_level  # 之前估计的噪声水平
            )

            # 记录指标到日志文件
            with open(metric_log_path, 'a') as f:
                if target_tensor is not None:
                    f.write(
                        f"{input_name}_denoise.png, {metrics['psnr']:.4f}, {metrics['ssim']:.4f}, "
                        f"{metrics['lpips']:.4f}, {metrics['niqe']:.4f}, {metrics['noise_level']:.4f}, "
                        f"{metrics['enhance_map']:.4f}, {metrics['target_map']:.4f}\n"
                    )
                else:
                    f.write(
                        f"{input_name}_denoise.png, N/A, N/A, N/A, {metrics['niqe']:.4f}, {metrics['noise_level']:.4f}, N/A, N/A\n")

            # 打印指标到控制台/日志
            if target_tensor is not None:
                logging.info(
                    f"Image {input_name} - PSNR: {metrics['psnr']:.4f}, SSIM: {metrics['ssim']:.4f}, "
                    f"LPIPS: {metrics['lpips']:.4f}, NIQE: {metrics['niqe']:.4f}, Noise: {metrics['noise_level']:.4f}, "
                    f"mAP(enhance): {metrics['enhance_map']:.4f}, mAP(target): {metrics['target_map']:.4f}"
                )
            else:
                logging.info(
                    f"Image {input_name} - NIQE: {metrics['niqe']:.4f}, Noise: {metrics['noise_level']:.4f} (no target provided)")


if __name__ == '__main__':
    main()