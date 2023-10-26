# NeuralCompression version 2
Bicephalous Convolutional Autoencoder for Time-Projection Chamber Data Compression, Version 2.

In this repo, we present code used for generating the results in the [paper](https://arxiv.org/abs/2310.15026)
_"Fast 2D Bicephalous Convolutional Autoencoder for Compressing 3D Time Projection Chamber Data"_
accepted to the "9th International Workshop on Data Analysis and Reduction for Big Scientific Data" ([DRBSD9](https://drbsd.github.io/))

## Set up running environment

### Directory
Download the repo to your local directory by running

```git clone https://github.com/BNL-DAQ-LDRD/NeuralCompression_v2.git```

And then, `cd` into the project folder `NeuralCompression_v2`. 
From now on, we assume `NeuralCompression_v2` is our current directory `./`, 
and all commands should be run inside `./`.

For simplicity, we assume that the data will be saved to `./data` and
the checkpoints will be saved to `./checkpoints`. **NOTE:** Please feel free to
save data and checkpoints to other locations, but don't forget to change the 
following commands accordingly.

### Conda environment package installation
Create the conda environment with the `yaml` file provided by running

```conda env create -f ./contrib/environment.yaml```

Activate the environment by running

```conda activate neuralcompress2```

Install the package by running

```python setup.py develop```

## Download data
We uploaded the data to [Zenodo](https://zenodo.org/records/10028587).

The data can be downloaded either directly from the website or by using the following command

```wget -P ./data https://zenodo.org/records/10028587/files/outer.tgz```

Decompress the dataset by

```tar -xvzf ./data/outer.tgz -C ./data```

More details of the data can be found in the paper and the Zenodo description.

## Download pretrained models

We uploaded the pretrained models to [Zenodo](https://zenodo.org/records/10028933).
We published three pretrained models:
- `BCAE-2D`: A 2D BCAE model with 4 encoder blocks and 8 decoder blocks
- `BCAE++`: A 3D BCAE model with a larger encoder and higher reconstruction accuracy
- `BCAE-HT`: A 3D BCAE model with a smaller encoder and higher throughput

The pretrained models can be downloaded either directly from the website or by using the following command:

```wget -P ./checkpoints https://zenodo.org/records/10028933/files/BCAEs.zip```

Decompress the pretrained models by

```unzip ./checkpoints/BCAEs.zip -d ./checkpoints```

### Note for pretrained models
1. All models were trained on log-scale ADC values, log2(ADC + 1);
2. All models were trained with transformation to regression output so that all values
   were above the (log) zero-suppression threshold.
3. More pretrained models for 2D `BCAE` models with different numbers of encoder and decoder
   blocks are available. We can share them upon request.

More details can be found in the [paper](https://arxiv.org/abs/2310.15026).

## Test
### Example test command
Run the following command to get the code and reconstruction result for 10 test examples.

```python train_test/test.py --data-path ./data/outer --checkpoint-path checkpoints/BCAE++  --save-path ./test --num-test-examples 10```

The `BCAE++` in the command can be replaced by `BCAE-HT` or `BCAE-2D`.

### Content of the result
In the folder `./test`, there will be a subfolder called `frames` and a `CSV` file called `metrics.csv`.
The `metrics.csv` contains a table with each row for one test example and metrics

| `occupancy` | `mse` | `mae` | `psnr` | `precision` | `recall` |

- `occupancy`: Fraction of non-zero voxels
- `mse`: Mean squared error of the reconstruction
- `mae`: Mean absolute error of the reconstruction
- `psnr`: [Peak signal-to-noise rate](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)

For the following two metrics, recall that a voxel will have label 1 if it has a positive ADC
value and 0 if otherwise, and the classification decoder of a `BCAE` model
will give a prediction on whether the voxel has a positive ADC value according to a threshold
(see `clf_threshold below).
- `precision`: Fraction of true 1 predictions among all 1 predictions. 
- `recall`: Frction of true 1 predictions among all voxels with label 1.
  


In the subfolder `frames`, we save one `NPZ` file for one input test example.
Each `NPZ` file has three fields:
- `input`: the input test example (if the model is trained the log-scale ADC values,
  the input is also saved in the log scale);
- `code`: the compressed data in half-precision;
- `reconstruction`: the reconstruction of the input.

### Other parameters for test
- `dimension`: the dimension of the data is loaded as.
  Use 2 for `BCAE-2D` model and use 3 for `BCAE++` and `BCAE-HT`. That is

  ```python train_test/test.py --data-path ./data/outer --dimension 2 --checkpoint-path ./checkpoints/BCAE-2D --save-path ./test```
  
  ```python train_test/test.py --data-path ./data/outer --dimension 3 --checkpoint-path ./checkpoints/BCAE-HT --save-path ./test```

  The `BCAE-HT` in the second command can be `BCAE++`, too.
  
- `log`: 0 for raw ADC value, 1 for log scale ADC value.
  (default = 1 since pretrained models were trained in log scale)
- `transform`: 0 for not using regression transformation, 1 for using transformation.
  (default = 1 since pretrained models were trained with the transformation)
- `clf-threshold`: classification threshold. A voxel will be masked with 0 if the
  classification output for the voxel is below the threshold, and 1 if otherwise.
- `device`: Device to run the test. Choose from `cpu` and `cuda`.
- `gpu-id`: The ID of the GPU card used.
- `half`: If the flag is used, we will use input and the encoder with half-precision.
  **NOTE**: The code will be saved in half-precision (float 16) no matter whether the code will be
  generated with half-precision or full-precision.
- `num-test-examples`: If you want to test on all existing test examples (18886),
  don't use the flag.

## Train
### Example train command
If you want to train models from scratch and 

- save the checkpoints to folder `./my_checkpoints`;
- train for 200 epochs with 100 epochs warmup (constant learning rate);

use the following commands:

- For 2D models:

  ```python train_test/train2d.py --data-path ./data/outer --num-epochs 200 --num-warmup-epochs 100 --checkpoint-path ./my_checkpoints```
- For 3D models:

  ```python train_test/train3d.py --data-path ./data/outer --num-epochs 200 --num-warmup-epochs 100 --checkpoint-path ./my_checkpoints```

### Other parameters for training
#### Shared parameters
For flag parameters `log`, `transform`, `clf-threshold`, `device`, and `gpu-id`, see Section [Other parameters for test](#other-parameters-for-test)

- `reg-loss`: Loss function used for evaluating reconstruction error.
  Choose from `mae` (mean absolute error) and `mse` (mean squared error).
- `half-training`: use the flag to turn on mixed-precision training
  (**Note:** I didn't notice any speedup, but I may not have implemented it correctly.
  Please let me know your experience with mixed-precision training.)
- `num-epochs`: Number of training epochs.
- `num-warmup-epochs`: Number of warmup epochs. It must be smaller than number of epochs.
  The number of epochs to keep the learning rate constant.
- `batches-per-epoch`: Number of batches trained in each epoch.
- `validation-batches-per-epoch`: Number of validation batches in each epoch.
- `sched-steps`: The steps for every decrease in learning rate.
  We use the `MultiStepLR` scheduler for the training.
  We will multiply the learning rate by a `gamma` < 1 every `sched-steps` steps
  after the first `num-warmup-epochs` epochs.
- `sched-gamma`: The learning rate is multiplied by gamma every `sched-steps` steps
  after passing `num-warmup-epochs` epochs.
- `batch-size`: Batch size.
- `learning-rate`: Learning rate.
- `save-frequency`: Saving checkpoints every `save-frequency` epochs.
- `checkpoint-path`: Directory to save checkpoints.

#### Model-specific parameters for 2D BCAEs (`train2d.py`)
- `num-encoder-layers`: Number of encoder blocks.
- `num-decoder-layers`: Number of decoder blocks.

#### Model-specific parameters for 3D BCAEs (`train3d.py`)
- `model-type`: Type of 3d BCAE models. Choose from (bcae++, bcae-ht).
  In this release, we provide two choices for 3d BCAE models:
  - `BCAE++`: A modification of the original BCAE;
  - `BCAE-HT`: A modification of BCAE++ by using smaller numbers of output channels in each encoder block.
    The HT here stands for high throughput.
