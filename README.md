# Learning-to-See-in-the-Dark
This is a Pytorch implementation of Learning to See in the Dark in CVPR 2018, by Chen Chen, Qifeng Chen, Jia Xu, and Vladlen Koltun.

This code includes the default model for training and testing on the See-in-the-Dark (SID) dataset.

# Setup
## Requirement for tensorflow
Required python (version 2.7) libraries: Tensorflow (>=1.1) + imageio + Numpy + Rawpy.

Tested in Granger server with Cuda (>=8.0) and CuDNN (>=5.0). 

## Requirement for pytorch
Required python (version 3.10) libraries: torch (>=2.1) + torchvision (>=0.16) + Numpy + Rawpy + imageio.

Tested in Lovegood server with Cuda (>=11.0). Minimum requirement GPU ram : 16 gb. CPU mode should also work but significantly slower.

## Dataset
You can download it directly from Google drive for the Sony(https://storage.googleapis.com/isl-datasets/SID/Sony.zip) (25 GB)

There is download limit by Google drive in a fixed period of time. If you cannot download because of this, try these links: Sony(https://drive.google.com/open?id=1G6VruemZtpOyHjOC5N8Ww3ftVXOydSXx) (25 GB).

In each row, there are a short-exposed image path, the corresponding long-exposed image path, camera ISO and F number. Note that multiple short-exposed images may correspond to the same long-exposed image.

The file name contains the image information. For example, in "10019_00_0.033s.RAF", the first digit "1" means it is from the test set ("0" for training set and "2" for validation set); "0019" is the image ID; the following "00" is the number in the sequence/burst; "0.033s" is the exposure time 1/30 seconds.

## Testing
Pretrained models can be found in checkpoint folder.
By default, the code takes the data in the "./dataset/Sony/" and the model from "./checkpoint/Sony/". If you save the dataset in other folders, please change the "input_dir" and "gt_dir" at the beginning of the code.

## Training new models
To train the Sony model, run "python train_Sony.py". The result and model will be save in "result_Sony" folder by default.

By default, the code takes the data in the "./dataset/Sony/" folder. If you save the dataset in other folders, please change the "input_dir" and "gt_dir" at the beginning of the code.

Loading the raw data and processing by Rawpy takes significant more time than the backpropagation. By default, the code will load all the groundtruth data processed by Rawpy into memory without 8-bit or 16-bit quantization. This requires at least 64 GB RAM for training the Sony model and 16 GB of GPU VRAM if you use the gpu implementation.

## Citation
Chen Chen, Qifeng Chen, Jia Xu, and Vladlen Koltun, "Learning to See in the Dark", in CVPR, 2018.