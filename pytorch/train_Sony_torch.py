import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.nn.parallel import DataParallel
import glob
import rawpy
import numpy as np
import time
import os
import imageio
import gc

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
    im = raw.raw_image_visible.astype(np.float16)
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

def preprocess_raw_images(train_ids, input_dir, gt_dir):
    processed_images = {}

    for train_id in train_ids:
        in_files = glob.glob(input_dir + f'{train_id:05d}_00*.ARW')
        in_path = in_files[np.random.randint(0, len(in_files) - 1)]
        in_fn = os.path.basename(in_path)

        gt_files = glob.glob(gt_dir + f'{train_id:05d}_00*.ARW')
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        raw = rawpy.imread(in_path)
        input_image = pack_raw(raw) * ratio

        gt_raw = rawpy.imread(gt_path)
        im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_image = np.float16(im / 65535.0)

        processed_images[train_id] = torch.from_numpy(input_image.transpose()), torch.from_numpy(gt_image.transpose())

    return processed_images

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, processed_images):
        self.processed_images = processed_images
        self.train_ids = list(processed_images.keys())

    def __len__(self):
        return len(self.train_ids)

    def __getitem__(self, idx):
        train_id = self.train_ids[idx]
        return self.processed_images[train_id]

def train(epoch, dataloader, model, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    start_time = time.time()

    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    end_time = time.time()
    epoch_time = end_time - start_time

    print('Epoch %d, Loss: %.3f, Time: %.2f seconds' % (epoch + 1, running_loss / len(dataloader), epoch_time))

def main():
    input_dir = '../dataset/Sony/short/'
    gt_dir = '../dataset/Sony/long/'
    checkpoint_dir = './result_Sony/'
    result_dir = './result_Sony/'
    train_fns = glob.glob(gt_dir + '0*.ARW')
    train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]
    save_freq = 30
    DEBUG = 1

    if DEBUG == 0:
        save_freq = 2
        train_ids = train_ids[0:5]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    processed_images = preprocess_raw_images(train_ids, input_dir, gt_dir)

    model = Network()

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = DataParallel(model)

    model.to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Load from checkpoint if available
    start_epoch = 0
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        criterion = checkpoint['loss']
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Create result directory if it does not exist
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    dataset = CustomDataset(processed_images)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=24)

    for epoch in range(start_epoch, 250):
        train(epoch, dataloader, model, criterion, optimizer, device)

        # Save checkpoint at every epoch
        checkpoint_path = os.path.join(result_dir, 'checkpoint.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
        }, checkpoint_path)

        with torch.no_grad():
            for ind in np.random.permutation(len(train_ids)):
                train_id = train_ids[ind]
                inputs, labels = dataset[ind]
                inputs = inputs.unsqueeze(0).to(device)
                labels = labels.unsqueeze(0).to(device)

                output = model(inputs)
                output = np.minimum(np.maximum(output.cpu().numpy(), 0), 1)

                if epoch % save_freq == 0:
                    epoch_result_dir = os.path.join(result_dir, f'{epoch:04d}')
                    if not os.path.exists(epoch_result_dir):
                        os.makedirs(epoch_result_dir)

                    print(output.shape)
                    temp = np.concatenate((labels.cpu().numpy()[0, :, :, :].T, output[0, :, :, :].T), axis=1)
                    imageio.imwrite(result_dir + '%04d/%05d_00_train.jpg' % (epoch, train_id), (temp * 255).astype(np.uint8))

if __name__ == "__main__":
    main()
