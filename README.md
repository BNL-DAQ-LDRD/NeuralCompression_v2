# NeuralCompression version 2
Bicephalous Convolutional Autoencoder for Time-Projection Chamber Data Compression, Version 2. 

[paper](https://arxiv.org/abs/2310.15026)

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

**Note for pretrained models**
1. All models were trained on log-scale ADC values, log2(ADC + 1);
2. All models were trained with transformation to regression output so that all values
   were above the (log) zero-suppression threshold.
3. More pretrained models for 2D BCAE models with different numbers of encoder and decoder
   blocks are available. Will update upon request.

## Test
As an example, assume that 
- the data is saved at `path_to_data`,
- the checkpoint folder is `path_to_checkpoints`,
- the result will be saved at `path_to_result`, and
- 10 examples will be tested.

Then one can run the following command to get the result
> `python train_test/test.py --data-path path_to_data --checkpoint-path path_to_checkpoints --save-path path_to_result --num-test-examples 10`

**Other parameters for `test.py`:**
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

**Result content**
In the `path_to_result`, there will be a folder called `frames` and a `CSV` file called `metrics.csv`. 
The `metrics.csv` will contain a table with columns `occupancy`, `mse`, `mae`, `psnr`, `precision`, and `recall`.
Each row of the table is for one input.

In the folder `frames`, we save the input, the code (compressed data in half precision), 
and the reconstruction of one input as an `NPZ` file with fields: `input`, `code`, and `reconstruction`.

## Train
If you want to train models from scratch, use the following commands
- For 2D models:
  > `python train_test/train2d.py --num-epochs 200 --num-warmup-epochs 100 --checkpoint-path path_to_checkpoints`
- For 3D models:
  > `python train_test/train3d.py --num-epochs 200 --num-warmup-epochs 100 --checkpoint-path path_to_checkpoints`

**Other parameters for `train2d.py`:** (TBD)
