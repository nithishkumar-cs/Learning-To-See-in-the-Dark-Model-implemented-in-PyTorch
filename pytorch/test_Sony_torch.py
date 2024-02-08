from __future__ import division
import os
import numpy as np
import rawpy
import glob
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

input_dir = './dataset/Sony/short/'
gt_dir = './dataset/Sony/long/'
checkpoint_dir = './checkpoint/Sony/'
result_dir = './result_Sony/'

# Get test IDs
test_fns = glob.glob(gt_dir + '/1*.ARW')
test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]

DEBUG = 0
if DEBUG == 1:
    save_freq = 2
    test_ids = test_ids[0:5]


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv11 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv13 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv14 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv15 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv16 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv17 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv18 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.conv19 = nn.Conv2d(32, 12, kernel_size=1, stride=1)

    def forward(self, x):

        conv1 = nn.functional.leaky_relu(self.conv1(x), 0.2)
        conv1 = nn.functional.leaky_relu(self.conv2(conv1), 0.2)
        pool1 = self.pool1(conv1)

        conv2 = nn.functional.leaky_relu(self.conv3(pool1), 0.2)
        conv2 = nn.functional.leaky_relu(self.conv4(conv2), 0.2)
        pool2 = self.pool2(conv2)

        conv3 = nn.functional.leaky_relu(self.conv5(pool2), 0.2)
        conv3 = nn.functional.leaky_relu(self.conv6(conv3), 0.2)
        pool3 = self.pool3(conv3)

        conv4 = nn.functional.leaky_relu(self.conv7(pool3), 0.2)
        conv4 = nn.functional.leaky_relu(self.conv8(conv4), 0.2)
        pool4 = self.pool4(conv4)

        conv5 = nn.functional.leaky_relu(self.conv9(pool4), 0.2)
        conv5 = nn.functional.leaky_relu(self.conv10(conv5), 0.2)

        up6 = self.upsample_and_concat(conv5, conv4, 256, 512)
        conv6 = nn.functional.leaky_relu(self.conv11(up6), 0.2)
        conv6 = nn.functional.leaky_relu(self.conv12(conv6), 0.2)

        up7 = self.upsample_and_concat(conv6, conv3, 128, 256)
        conv7 = nn.functional.leaky_relu(self.conv13(up7), 0.2)
        conv7 = nn.functional.leaky_relu(self.conv14(conv7), 0.2)

        up8 = self.upsample_and_concat(conv7, conv2, 64, 128)
        conv8 = nn.functional.leaky_relu(self.conv15(up8), 0.2)
        conv8 = nn.functional.leaky_relu(self.conv16(conv8), 0.2)

        up9 = self.upsample_and_concat(conv8, conv1, 32, 64)
        conv9 = nn.functional.leaky_relu(self.conv17(up9), 0.2)
        conv9 = nn.functional.leaky_relu(self.conv18(conv9), 0.2)

        conv10 = self.conv19(conv9)
        out = nn.functional.pixel_shuffle(conv10, 2)
        # print(out.size())
        return out

    def upsample_and_concat(self, x1, x2, output_channels, in_channels):
        # print(f"x1: {x1.size()} x2: {x2.size()}")
        pool_size = 2
        deconv_filter = torch.nn.Parameter(torch.randn(in_channels, output_channels, pool_size, pool_size) * 0.02)
        deconv_filter = deconv_filter.to(x1.device)
        upsampled = F.conv_transpose2d(x1, deconv_filter, stride=pool_size)
        concatenated = torch.cat([upsampled, x2], dim=1)
        # print(concatenated.size())
        return concatenated

def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


model = Network()
checkpoint = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pth'))

# Check if the model was trained with DataParallel
if 'module.' in list(checkpoint['model_state_dict'].keys())[0]:
    # Create a new state dictionary without the 'module.' prefix
    new_state_dict = {k[7:]: v for k, v in checkpoint['model_state_dict'].items()}
else:
    # Use the original state dictionary
    new_state_dict = checkpoint['model_state_dict']

# Load the state dictionary
model.load_state_dict(new_state_dict)

model.eval()

if not os.path.isdir(result_dir + 'final/'):
    os.makedirs(result_dir + 'final/')

for test_id in test_ids:
    in_files = glob.glob(input_dir + '%05d_00*.ARW' % test_id)
    for k in range(len(in_files)):
        in_path = in_files[k]
        in_fn = os.path.basename(in_path)
        gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % test_id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        raw = rawpy.imread(in_path)
        input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio

        im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

        gt_raw = rawpy.imread(gt_path)
        im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

        input_full = np.minimum(input_full, 1.0)

        input_tensor = torch.from_numpy(input_full).permute(0, 3, 1, 2).float()
        output_tensor = model(input_tensor)
        output = output_tensor.detach().permute(0, 2, 3, 1).numpy()
        output = np.minimum(np.maximum(output, 0), 1)

        output = output[0, :, :, :]
        gt_full = gt_full[0, :, :, :]
        scale_full = scale_full[0, :, :, :]
        scale_full = scale_full * np.mean(gt_full) / np.mean(scale_full)

        imageio.imwrite(result_dir + 'final/%5d_00_%d_out.png' % (test_id, ratio), (output * 255).astype(np.uint8))
        imageio.imwrite(result_dir + 'final/%5d_00_%d_scale.png' % (test_id, ratio), (scale_full * 255).astype(np.uint8))
        imageio.imwrite(result_dir + 'final/%5d_00_%d_gt.png' % (test_id, ratio), (gt_full * 255).astype(np.uint8))
