# import torch
# from torchvision.transforms import functional as F
# from data import valid_dataloader
# from utils import Adder
# import os
# from skimage.metrics import peak_signal_noise_ratio
# import torch.nn.functional as f
#
#
# def _valid(model, args, ep):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     its = valid_dataloader(args.data_dir, batch_size=1, num_workers=0)
#     model.eval()
#     psnr_adder = Adder()
#
#     with torch.no_grad():
#         print('Start Evaluation')
#         factor = 8
#         for idx, data in enumerate(its):
#             input_img, label_img = data
#             input_img = input_img.to(device)
#
#             h, w = input_img.shape[2], input_img.shape[3]
#             H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
#             padh = H-h if h%factor!=0 else 0
#             padw = W-w if w%factor!=0 else 0
#             input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')
#
#             if not os.path.exists(os.path.join(args.result_dir, '%d' % (ep))):
#                 os.mkdir(os.path.join(args.result_dir, '%d' % (ep)))
#
#             pred = model(input_img)[2]
#             pred = pred[:,:,:h,:w]
#
#             pred_clip = torch.clamp(pred, 0, 1)
#             p_numpy = pred_clip.squeeze(0).cpu().numpy()
#             label_numpy = label_img.squeeze(0).cpu().numpy()
#
#             psnr = peak_signal_noise_ratio(p_numpy, label_numpy, data_range=1)
#
#             psnr_adder(psnr)
#             print('\r%03d'%idx, end=' ')
#
#     print('\n')
#     model.train()
#     return psnr_adder.average()
import torch
from torchvision.transforms import functional as F
from data import valid_dataloader
from utils import Adder
import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim
import torch.nn.functional as f
import numpy as np


def _valid(model, args, ep):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    its = valid_dataloader(args.data_dir, batch_size=1, num_workers=0)
    model.eval()
    psnr_adder = Adder()
    ssim_adder = Adder()

    with torch.no_grad():
        print('Start Evaluation')
        factor = 8
        for idx, data in enumerate(its):
            input_img, label_img = data
            input_img = input_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor * factor)
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')

            if not os.path.exists(os.path.join(args.result_dir, '%d' % (ep))):
                os.mkdir(os.path.join(args.result_dir, '%d' % (ep)))

            pred = model(input_img)[2]
            pred = pred[:, :, :h, :w]

            pred_clip = torch.clamp(pred, 0, 1)
            p_numpy = pred_clip.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # 形状调整为 (H, W, C)
            label_numpy = label_img.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # 形状调整为 (H, W, C)

            # 计算PSNR
            psnr = peak_signal_noise_ratio(p_numpy, label_numpy, data_range=1)
            psnr_adder(psnr)

            # 计算SSIM
            win_size = min(h, w, 7)  # 动态调整窗口大小，确保不超出图像尺寸
            ssim_value = ssim(p_numpy, label_numpy, win_size=win_size, channel_axis=2, data_range=1)
            ssim_adder(ssim_value)

            print(f'\r{idx:03d} PSNR: {psnr:.4f} SSIM: {ssim_value:.4f}', end=' ')

    print('\n')
    model.train()

    # 返回 PSNR 和 SSIM 的平均值
    avg_psnr = psnr_adder.average()
    avg_ssim = ssim_adder.average()

    print(f'Average PSNR: {avg_psnr:.4f}, Average SSIM: {avg_ssim:.4f}')

    return avg_psnr, avg_ssim
