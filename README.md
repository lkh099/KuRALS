# KuRALS: Ku-Band Radar Datasets for Multi-Scene Long-Range Surveillance with Baselines and Loss Design

## Introduction to the KuRALS dataset

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
- Implementations of multiple baseline segmentation models (`kuralsnet` and variants, FCN8s, U-Net, DeepLabv3+, HRNet, RSSNet, SegFormer, Swin Transformer) along with their training configurations;
- Implementations of different loss functions;
- `KuRALSNetNPUSeg`: an NPU-legal (128-MAC-array-deployable) segmentation model with quantization-aware training and an int8 export pipeline;
- A Kalman-filter + Hungarian-assignment multi-target tracker built on top of the NPU model's per-frame detections.

See [`docs/USAGE.md`](docs/USAGE.md) for how to train, evaluate, run the tracker demo, and export a model for NPU deployment. See [`CLAUDE.md`](CLAUDE.md) for a summary of the current model status and what's been tried and ruled out.

## Installation

This project runs from a plain Python virtualenv (not conda), on Python 3.10 with torch 2.3.1+cu121 / torchvision 0.18.1+cu121 -- the versions this codebase is actually developed and tested against.

1. Clone this repository and bootstrap the environment:
```bash
$ git clone https://github.com/lkh099/KuRALS
$ cd KuRALS/
$ bash scripts/setup_env.sh          # creates .venv/, installs torch (cu121) + requirements.txt + this package (editable)
$ source .venv/bin/activate
```
If your GPU/driver needs a different CUDA build, edit the `--index-url` in `scripts/setup_env.sh` (see [pytorch.org/get-started/previous-versions](https://pytorch.org/get-started/previous-versions/)).

2. (Optional) Build the `correlation` CUDA extension, only needed for the `kuralsnet_adapkcxi` baseline variant (unrelated to the NPU/tracker work):
```bash
$ pip install ninja
$ cd kurals/correlation
$ python setup.py install
```
Assign your venv's Python include path (e.g. `.venv/lib/python3.10/site-packages` or the interpreter's own include dir) to `include_dirs` in `kurals/correlation/setup.py` first if the build fails to find `Python.h`.

3. Point the repo at your dataset and log directories:
```bash
$ cd kurals/utils/
$ python set_paths.py --cwr /path/to/KuRALS_CW --pdr /path/to/KuRALS_PD --logs /path/to/logs
```
This writes `kurals/config_files/config.ini`, which every training/eval script reads.

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

## Acknowledgements
- We thank the authors of [MVRSS](https://arxiv.org/abs/2103.16214) for providing the basic codebase upon which part of this repository is built.
- Users of this dataset are kindly requested to acknowledge that the data were collected by the Intelligent Science and Technology Academy of CASIC.

## To-Do List

- [ ] Supplement `annotations.json` with additional information, including timestamps and scan mode.
