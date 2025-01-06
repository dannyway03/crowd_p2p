# P2PNet-based Croud Counting

 PyTorch implementation of **P2PNet** as described in [Rethinking Counting and Localization in Crowds: A Purely Point-Based Framework](https://arxiv.org/abs/2107.12746).

# Installation (Ubuntu 22.04)

1. Clone this repository.
   ```Shell
   git clone git@github.com:dannyway03/crowd_p2p.git
   cd crowd_p2p
   ```
   
2. Create python venv (tested on python 3.10)
   ```shell
   python3 -m venv venv
   source venv/bin/activate
   ```
   
3. Install Torch == v2.5.1 following official instruction \
   Modify according your platform (cuda/cpu):
   ```shell
   pip3 install torch=2.5.1 torchvision=0.20.1 --index-url https://download.pytorch.org/whl/<cu102, cu111, cpu>
   ```
4. Install package dependencies
   ```shell
   pip3 install -r requirements.txt
   ```

5. Download and organize your datasets (optional, for training)
    * Download ShanghaiTechCounting dataset (https://www.kaggle.com/datasets/xyyu18/shanghaitech-crowd-counting-dataset?resource=download)
    * Unzip to DATASET_ROOT
    * Organize the dataset, e.g., 
      ```shell
      python prepare_dataset.py [DATASET_ROOT] [DATASET_TYPE] [DATA_ROOT]
      ```

# Ccounting dataset
We use a list file to collect all the images and their ground truth annotations in a counting dataset. When your dataset is organized as recommended in the following, the format of this list file is defined as:
```
train/scene01/img01.jpg train/scene01/img01.txt
train/scene01/img02.jpg train/scene01/img02.txt
...
train/scene02/img01.jpg train/scene02/img01.txt
```

## Dataset structures:
```
DATA_ROOT/
        |->train/
        |    |->scene01/
        |    |->scene02/
        |    |->...
        |->test/
        |    |->scene01/
        |    |->scene02/
        |    |->...
        |->train.list
        |->test.list
```
DATA_ROOT is your path containing the counting datasets.

### Annotations format
For the annotations of each image, we use a single txt file which contains one annotation per line. Note that indexing for pixel values starts at 0. The expected format of each line is:
```
x1 y1
x2 y2
...
```

## Training

The network can be trained using the `train.py` script. For training on SHTechPartA, use

```
CUDA_VISIBLE_DEVICES=0 python train.py --data_root $DATA_ROOT \
    --dataset_file SHHA \
    --epochs 3500 \
    --lr_drop 3500 \
    --output_dir ./logs \
    --checkpoints_dir ./weights \
    --tensorboard_dir ./logs \
    --lr 0.0001 \
    --lr_backbone 0.00001 \
    --batch_size 8 \
    --eval_freq 1 \
    --gpu_id 0
```
By default, a periodic evaluation will be conducted on the validation set.

## Testing

A trained model (with an MAE of **51.96**) on SHTechPartA is available at "./weights", run the following commands to launch a visualization demo:

```
CUDA_VISIBLE_DEVICES=0 python run_test.py --weight_path ./weights/SHTechA.pth --output_dir ./logs/
```