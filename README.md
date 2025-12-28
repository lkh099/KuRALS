# KuRALS: Ku-Band Radar Datasets for Multi-Scene Long-Range Surveillance with Baselines and Loss Design

## Updates

- 12/2025 The KuRALS dataset and the associated data-processing code are publicly released.

## Introductions of KuRALS dataset

KuRALS is a range-Doppler (RD)-level radar surveillance dataset designed for learning-based long-range detection of moving targets. The dataset covers aerial (unmanned aerial vehicles), land (pedestrians and cars) and maritime (boats) scenarios. It is real-measured by two long-range Kurz-under (Ku) band radars and contains two subsets (KuRALS-CW and KuRALS-PD). It consists of RD spectrograms with pixel-wise annotations of categories, velocity and range coordinates, and the azimuth and elevation angles are also provided.

The KuRALS dataset is publicly available at the following link: [Google Drive](https://drive.google.com/drive/folders/1f5GEZdeYtSD2hMt0ruKV8FECTGHc_crd?usp=drive_link). The raw RD data can also be accessed here: [Baidu Netdisk](https://pan.baidu.com/s/1f9z1JwCgqMAP2by_Vpf33g?pwd=1uu4). The structures of both the KuRALS-CW and KuRALS-PD datasets are summarized as follows:

```
data_root
  - <SEQ_NAME>
  | - annotations
  | | - box
  | | | - range_doppler_light.json
  | | - dense
  | | | - <FRAME_ID>
  | | | | - range_doppler.npy
  | | - sparse
  | | | - <FRAME_ID>
  | | | | - range_doppler.npy
  | - range_doppler_numpy
  |   - <FRAME_ID>.npy
  - annotations.json
  - data_seq_ref.json
  - light_dataset_frame_oriented.json
  - rd_stats_all.json
  - rd_weights.json
  - sequence.txt
```

Additional information about each target, such as azimuth, elevation and energy, is provided in the `annotations.json` file. Its structure is as follows:

```
- <SEQ_NAME>
| - <FRAME_ID>
| | - <tgtType>
| | - <scPeriod>
| | - <frmID>
| | - <Distance>
| | - <Velocity>
| | - <Azimuth>
| | - <Height>
| | - <EleBeamIndex>
| | - <energy>
| | - <scr>
| | - <rdCoord>
| | - <rdBox>
| | - <rdDense>
```

## Basic description of this code

This repository provides a complete end-to-end pipeline for the KuRALS dataset, covering data processing, label generation and algorithm validation. Specifically, it includes:
- Processing complex-valued RD data to generate RD inputs for model training;
- Automatic label refinement and generation by jointly leveraging CFAR, NMS and related techniques;
- Implementations of multiple models along with their corresponding training configurations;
- Implementations of different loss functions;
- Training and evaluation scripts.

## Installation

We provide instructions on how to install dependencies via conda and pip:

1. Create and activate a new conda environment:
```bash
$ conda create -n kurals python=3.8
$ conda activate kurals
```

2. Git clone this repository and install it using pip:
```bash
$ git clone https://github.com/lihua199710/KuRALS
$ cd KuRALS/
$ pip install -e .
```
With this, you can edit the KuRALS code on the fly and import function and classes of KuRALS in other projects as well.

3. Install pytorch using conda.
```bash
$ conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

4. (Optional) Install correlation package for the usage of KuRALS-Net w/ $\text{AdaPKC}^{\xi}$: First assign the python path of the `kurals` conda environment, such as `home/miniconda/envs/kurals/include/python3.8`, to `include_dirs` in [kurals/correlation/setup.py](./kurals/correlation/setup.py). Then run the following command lines:
```bash
$ pip install ninja
$ cd kurals/correlation
$ python setup.py install
```

5. Install other dependencies using pip.
```bash
$ pip install -r requirements.txt
```

6. (Optional) To uninstall this package, run:
```bash
$ pip uninstall kurals
```

## Usage

In any case, it is **mandatory** to specify beforehand both the path where the Radar dataset is located and the path to store the logs and models. For example: I put the Ku-band Radar folder in /home/datasets_local, the path I should specify is /home/datasets_local. The same way if I store my logs in /home/logs. Please run the following command lines while adapting the paths to your settings:

```bash
$ cd kurals/utils/
$ python set_paths.py --cwr /home/datasets_local/KuRALS_CW --pdr /home/datasets_local/KuRALS_PD --logs /home/logs
```

### Prepare the KuRALS dataset

#### (Option 1) Download the KuRALS dataset

The KuRALS dataset is accessible via this link: [Google Drive](https://drive.google.com/drive/folders/1f5GEZdeYtSD2hMt0ruKV8FECTGHc_crd?usp=drive_link)

#### (Option 2) Download the complex-valued RD data and generate the KuRALS dataset

The raw RD data can be accessed via this link: [Baidu Netdisk](https://pan.baidu.com/s/1f9z1JwCgqMAP2by_Vpf33g?pwd=1uu4). We provide dataset generation pipelines for both the KuRALS-CW and KuRALS-PD datasets, which can be executed using the [dataset_generate.sh](./kurals/dataset_process/dataset_generate.sh) script. Before running this script, please ensure the following steps are completed:

1. Replace the corresponding dataset processing function in `dataset_generate.sh`. Specifically, use `kuralscw_processing.py` for KuRALS-CW and `kuralspd_processing.py` for KuRALS-PD.

2. Specify the path to the complex-valued RD data (via the `data-path` variable) and the output path for the generated dataset (via the `save-path` variable) in `dataset_generate.sh`.

After completing the above two steps, the dataset can be generated by executing:
```bash
$ cd kurals/dataset_process/
$ bash dataset_generate.sh
```

### Training

#### 1. Model Specification

To train a model, a JSON configuration file must be specified. We provide configuration files for KuRALS-Net and its related variants, as listed below:
- `kurals/config_files/kuralsnet.json`: KuRALS-Net
- `kurals/config_files/kuralsnet_woaspp.json`: KuRALS-Net without the ASPP module
- `kurals/config_files/kuralsnet_ada.json`: KuRALS-Net with the ADA module (from TransRadar)
- `kurals/config_files/kuralsnet_pkc.json`: KuRALS-Net with the PKC module
- `kurals/config_files/kuralsnet_adapkcxi.json`: KuRALS-Net with the AdaPKC-Xi module
- `kurals/config_files/kuralsnet_adapkctheta.json`: KuRALS-Net with the AdaPKC-Theta module

In addition, we also provide configuration files for other architectures, including FCN8s, U-Net, DeepLabv3+, HRNet, RSSNet, SegFormer and Swin Transformer. 

#### 2. Training Loss Specification

We provide multiple loss functions, including CE loss, weighted CE (wCE) loss, Focal loss, Dice loss, Generalized Dice loss and NBS loss. The desired loss function can be selected by setting the `custom_loss` variable in the above model configuration file. The corresponding mapping is defined in the `define_loss` function in `kurals/utils/functions.py`.

#### 3. Dataset Specification

The supported datasets include KuRALS-CW and KuRALS-PD. Different datasets can be selected by passing the `--dataset` argument in the training script [train.sh](./kurals/train.sh).

#### 4. Training Execution

For example, to train the KuRALS-Net architecture with the $\text{NBS}^{1}$ loss on the KuRALS-CW dataset, please run the following command lines:

```bash
$ cd KuRALS/kurals
$ bash train.sh
```

### Testing

To test a recorded model, you should specify the configuration file and the path of model weights. For example, if you want to test the KuRALS-Net model and the model weights have been saved to `KuRALS/test_results/kuralsnet_cw.pt`, you should assign this path to `--model-path` in [test.sh](./kurals/test.sh). This way, you should execute the following command lines:

```bash
$ cd KuRALS/kurals
$ bash test.sh
```

As a reference, we provide pretrained KuRALS-Net model weights trained with the NBS loss on the KuRALS-CW and KuRALS-PD datasets. The corresponding checkpoints can be found at `test_results/kuralsnet_cw.pt` and `test_results/kuralsnet_pd.pt`, respectively.

### More Evaluations

#### Computational Complexity and Runtime

The computational complexity, GPU memory consumption and inference speed of different models can be evaluated by running the following script:
```bash
$ cd KuRALS/kurals
$ bash flops_fps.sh
```

#### Comparision with CFAR

The foreground–background segmentation performance of CFAR on the KuRALS-CW or KuRALS-PD datasets can be evaluated by running the following script:
```bash
$ cd KuRALS/kurals
$ bash test_cfar.sh
```

## Acknowledgements
- We thank the authors of [MVRSS](https://arxiv.org/abs/2103.16214) for providing the basic codebase upon which part of this repository is built.
- Users of this dataset are kindly requested to acknowledge that the data were collected by the Intelligent Science and Technology Academy of CASIC.

## To-Do List

- [ ] Supplement `annotations.json` with additional information, including timestamps and scan mode.