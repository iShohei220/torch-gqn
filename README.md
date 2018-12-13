# PyTorch implementation of Generative Query Network
### Original Paper: Neural scene representation and rendering (Eslami, et al., 2018)
#### https://deepmind.com/blog/neural-scene-representation-and-rendering

![img](https://storage.googleapis.com/deepmind-live-cms/images/model.width-1100.png)

#### Pixyz Implementation: https://github.com/masa-su/pixyzoo/tree/master/GQN

## Requirements
- Python >=3.6
- PyTorch
- TensorBoardX

## How to Train
```
python train.py --train_data_dir /path/to/dataset/train --test_data_dir /path/to/dataset/test

# Using multiple GPUs.
python train.py --device_ids 0 1 2 3 --train_data_dir /path/to/dataset/train --test_data_dir /path/to/dataset/test
```
### ❗️❗️CAUTION❗️❗️
#### This implementation needs very high computational power because of enormous number of parameters.
#### The default setting is based on the setting on the original GQN paper (In the paper, 4x GPUs with 24GB memory are used in the experiments).
#### If you have only limited GPU memory, I recommend to use the option of `shared_core True` (default: False) or `layers 8` (default 12) to reduce parameters.
#### As far as I experimented, this change would not affect the quality of results so much, although the setting would be different with the original paper.

## Dataset
#### https://github.com/deepmind/gqn-datasets

## Usage
### dataset/convert2torch.py
Convert TFRecords of the dataset for PyTorch implementation.

### representation.py
Representation networks (See Figure S1 in Supplementary Materials of the paper).

### core.py
Core networks of inference and generation (See Figure S2 in Supplementary Materials of the paper).

### conv_lstm.py
Implementation of convolutional LSTM used in `core.py`.

### gqn_dataset.py
Dataset class.

### model.py
Main module of Generative Query Network.

### train.py
Training algorithm.

### scheduler.py
Scheduler of learning rate used in `train.py`.

## Results (WIP)
||Ground Truth|Generation|
|---|---|---|
|Shepard-Metzler objects|![shepard_ground_truth](https://user-images.githubusercontent.com/24241353/49865725-100aa180-fe49-11e8-9ae4-cd9ed54a6bc2.png)|![shepard_generation](https://user-images.githubusercontent.com/24241353/49865970-bb1b5b00-fe49-11e8-9ce3-264476022045.png)|
|Mazes|![mazes_ground_truth](https://user-images.githubusercontent.com/24241353/49866239-8d82e180-fe4a-11e8-8f1d-038c922686a0.png)|![mazes_generation](https://user-images.githubusercontent.com/24241353/49866241-8eb40e80-fe4a-11e8-92c2-11de1bb0407d.png)|

