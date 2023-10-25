# NeuralCompression version 2
Bicephalous Convolutional Autoencoder for Time-Projection Chamber Data Compression, Version 2.

In this repo, we present code used for generating the results in [paper](https://arxiv.org/abs/2310.15026)
_"Fast 2D Bicephalous Convolutional Autoencoder for Compressing 3D Time Projection Chamber Data"_
accepted to the "9th International Workshop on Data Analysis and Reduction for Big Scientific Data" ([DRBSD9](https://drbsd.github.io/))

## Setup running environment

Create the conda environment with the `yaml` file provided by running
> `conda env create -f contrib/environment.yaml`

Activate the environment by running
> `conda activate neuralcompress2`

Install the package by running
> `python setup.py develop`

## Data
We uploaded the data to [Zenodo](https://zenodo.org/records/10028587).

The data can be downloaded either directly from the website or by using the following command
> `wget https://zenodo.org/records/10028587/files/outer.tgz`

Decompress the dataset by
> `tar -xvzf outer.tgz`

More details of the data can be found in the paper and the Zenodo description.

## Pretrained Models

We uploaded the pretrained models to [Zenodo](https://zenodo.org/records/10028933).
We published three pretrained models:
- `BCAE-2D`: A 2D BCAE model with 4 encoder blocks and 8 decoder blocks
- `BCAE++`: A 3D BCAE model with a larger encoder and higher reconstruction accuracy
- `BCAE-HT`: A 3D BCAE model with a smaller encoder and higher throughput

The pretrained models can be downloaded either directly from the website or by using the following command:
> `wget https://zenodo.org/records/10028933/preview/BCAEs.zip`

Decompress the pretrained models by
> `unzip BCAEs.zip`

**NOTE:** Do make sure to use `md5sum [FILENAME]` to check whether the `md5` checksum matches with
the ones listed under the Zenodo download link
![md5checksum](https://github.com/BNL-DAQ-LDRD/NeuralCompression_v2/assets/22546248/673eb86a-3228-450e-bb16-beedb16adb44)


### Note for pretrained models
1. All models were trained on log-scale ADC values, log2(ADC + 1);
2. All models were trained with transformation to regression output so that all values
   were above the (log) zero-suppression threshold.
3. More pretrained models for 2D BCAE models with different numbers of encoder and decoder
   blocks are available. Will update upon request.

## Test
### Example test command
Assume that
- the data is saved at `path_to_data`,
- the checkpoint folder is `path_to_checkpoints`,
- the result will be saved at `path_to_result`, and
- 10 examples will be tested.

Then, one can run the following command inside the folder `train_test` to get the result
> `python test.py --data-path path_to_data --checkpoint-path path_to_checkpoints --save-path path_to_result --num-test-examples 10`

### Content of the result
In the `path_to_result`, there will be a folder called `frames` and a `CSV` file called `metrics.csv`.
The `metrics.csv` will contain a table with columns `occupancy`, `mse`, `mae`, `psnr`, `precision`, and `recall`.
Each row of the table is for one input.

In the folder `frames`, we save the input, the code (compressed data in half precision),
and the reconstruction of one input as an `NPZ` file with fields: `input`, `code`, and `reconstruction`.

### Other parameters for test
- `dimension`: the dimension of the data is loaded as. Use 2 for BCAE-2D model and use 3 for BCAE++ and BCAE-HT
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
If you want to train models from scratch, use the following commands inside of the folder `train_test`
- For 2D models:
  > `python train2d.py --data-path path_to_data --num-epochs 200 --num-warmup-epochs 100 --checkpoint-path path_to_checkpoints`
- For 3D models:
  > `python train3d.py --data-path path_to_data --num-epochs 200 --num-warmup-epochs 100 --checkpoint-path path_to_checkpoints`

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
- `sched-gamma`: The gamma that is multiplied to the learning rate.
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
