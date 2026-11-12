# PCB-SPMF

RGB–hyperspectral component detection with **Safe Positive Modulation Fusion (SPMF)**. This repository provides spectral construction, spatial registration, RGB pretraining, frozen RGB–HSI adaptation, checkpoint evaluation, and paired statistics.

## Installation

Use separate environments for spectral preprocessing and CUDA detection.

**Spectral preprocessing — Python 3.10:**

```bash
python -m pip install -r environments/preprocessing_requirements.txt
```

**Detection — Linux, Python 3.8, CUDA 11.7:**

```bash
python -m pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --index-url https://download.pytorch.org/whl/cu117
python -m pip install -r environments/training_requirements.txt
```

The reference runtime uses an RTX 3090 and **cuDNN 8.8.1 (8801)**. Install that cuDNN version and place its libraries before other CUDA libraries when starting Python:

```bash
export LD_LIBRARY_PATH=/path/to/cudnn-8.8.1/lib:/path/to/cuda-11.7/lib64:$LD_LIBRARY_PATH
```

The entry point imports the included Ultralytics source snapshots. Do not replace them with a separately installed Ultralytics implementation. Evaluation checks the PyTorch/cuDNN versions and uses CUDA FP16. Statistical replay requires only NumPy and SciPy.

## Dataset

Download **PCB-Vision, Version 1** from the [official dataset page](https://doi.org/10.14278/rodare.2704). Extract `PCBDataset.zip` and locate the directory containing `RGB/` and `HSI/`:

```text
PCBDataset/
├── RGB/
│   ├── 1.jpg ... 53.jpg
│   └── ...
└── HSI/
    ├── pcb1/ ... pcb53/
    ├── PCB_Masks/
    └── Monoseg_mask/
```

**Neither raw dataset images nor processed spectral images are included in the repository or release assets.** The repository includes the experiment's 53 YOLO bounding-box annotation files, the fixed 42/11 board split, and manual registration parameters. These annotations are provided separately because the downloaded semantic masks do not directly specify the experiment's detection boxes. Classes are `0: ic`, `1: capacitor`, and `2: connector`.

In the spectral environment, from the repository root:

```bash
python -X utf8 reproduce.py prepare-data --raw-root /path/to/PCBDataset
```

This copies the original RGB files without re-encoding, installs the supplied detection annotations, applies the saved LDA transforms, and performs manual registration followed by the original image-based ECC refinement. It creates `data/images/`, `data/labels/`, and `data/spectral/`. Existing matching files are reused; different files are not overwritten. Generated inputs are compared with the expected checksums. Temporary projection and registration outputs remain under `data/build_*/`.

To prepare only one experiment, use `--profile main` or `--profile paired`. The default `all` also generates the separate timing input: bands `[155, 88, 20]` (zero-based), per-image/channel 2%–98% scaling, and the original manual transform. That timing representation is distinct from the LDA representation used for AP evaluation.

To fit the spectral transformations from the original data, rather than apply the supplied transforms:

```bash
python -X utf8 reproduce.py verify-raw --raw-root /path/to/PCBDataset
python -X utf8 reproduce.py fit-spectral --profile main --raw-root /path/to/PCBDataset --output outputs/spectral_main
python -X utf8 reproduce.py fit-spectral --profile paired --raw-root /path/to/PCBDataset --output outputs/spectral_paired
```

Band scoring uses minimum class-pair separability, excludes 20 bands at each edge, and applies a minimum band spacing of 15 and a correlation threshold of 0.98. LDA produces three channels with fitted 1%–99% normalization. The maximum selected-band count is six; the recorded training-only fit retains five bands under the same screening rules. Parameters and fitted transforms are in `configs/` and `transforms/`.

## SPMF and frozen adaptation

SPMF uses a learned Sigmoid reliability factor, channel and spatial gates, and a positive modulation cap of **0.50**. The reliability predictor remains trainable throughout adaptation, with **no training-time reliability floor**. Its output weights start at zero and its bias at -0.5. RGB backbone, neck, head, and batch-normalization parameters and running statistics remain fixed. The implementation settings are in `configs/spmf.json`.

Frozen adaptation uses AdamW for 120 epochs, batch size 8, base learning rate 0.0003, gate learning rate 0.002, and base weight decay 0.0001 (zero for gates). The initial tensors are loaded before applying the same SPMF settings to both the model and its EMA copy.

## Checkpoints and results

Download [PCB-SPMF-assets-v1.1.0.zip](https://github.com/Xu-Hanlin/PCB-SPMF/releases/download/v1.1.0/PCB-SPMF-assets-v1.1.0.zip) and extract it into the repository root:

```bash
python -m zipfile -e /path/to/PCB-SPMF-assets-v1.1.0.zip .
python reproduce.py verify
```

The asset contains the manuscript checkpoints, unadapted initialization states, reference detection records, training curves, and paired bootstrap samples. Dataset images and HSI cubes are not included. Before installing assets, `python reproduce.py verify --code-only` checks the source files alone.

| Evaluation | RGB mAP50:95 | SPMF mAP50:95 | Mean difference (percentage points) |
|---|---:|---:|---:|
| Main, three adaptation seeds sharing one RGB prior | 0.635033 | 0.638234 | +0.320110 |
| Paired, fourteen independently pretrained RGB priors | 0.616832 | 0.620324 | +0.349236 |

The main profile applies the supervised LDA transform fitted on all 53 boards. The paired profile fits spectral processing on the 42 training boards only. Both use the 42/11 detector split. Main adaptation seeds are 0, 1, and 2; paired RGB initialization seeds are 3–16. Main seeds share one initial state and are not three independently pretrained RGB detectors.

For the fourteen pairs, the mAP50:95 training-seed 95% interval is [0.222354, 0.476117] percentage points and the paired-PCB interval is [0.179472, 0.425240]. These describe the specified validation protocol; validation boards also supply checkpoint selection. Full-precision P, R, AP50, AP50:95 and all per-seed results are provided in `reference/`.

## Checkpoint evaluation

Run in the CUDA environment after preparing the data and installing the assets:

```bash
python reproduce.py evaluate --profile main --seeds all --output outputs/main_evaluation
python reproduce.py evaluate --profile paired --seeds all --output outputs/paired_evaluation
```

Use `--seeds 0` for one main checkpoint or `--seeds 3,4` for a paired subset. Evaluation uses batch 16, native rectangular padding, nominal size 640, FP16, unfused Conv/BN, confidence 0.001, NMS IoU 0.70, and at most 300 detections. The validation canvas is 416×672. Each checkpoint is checked against its saved metrics and native detection records. SPMF OFF must recover its corresponding RGB predictions exactly.

## Paired statistics

```bash
python reproduce.py replay-statistics --output outputs/statistics
python reproduce.py replay-statistics --bootstrap --output outputs/statistics_bootstrap
```

Both commands recompute pooled AP and seed statistics from the supplied board records. `--bootstrap` additionally recomputes all 2,000 shared paired PCB resamples; without it, the saved bootstrap differences supply the PCB intervals. Resampling uses the same boards for RGB and SPMF and all seeds, with pooled AP recomputed before averaging differences. These commands perform no network inference.

## Training

```bash
python reproduce.py train-rgb --seed 3 --output outputs/rgb_seed_03
python reproduce.py train-frozen --profile main --seed 0 --output outputs/main_seed_00
python reproduce.py train-frozen --profile main --seed 1 --output outputs/main_seed_01
python reproduce.py train-frozen --profile main --seed 2 --output outputs/main_seed_02
python reproduce.py train-frozen --profile paired --seed 3 --output outputs/paired_seed_03
```

RGB pretraining uses 500 epochs. Each `train-frozen` command starts from its supplied unadapted state. It does not automatically consume a preceding `train-rgb` output; migration of a new RGB prior is implemented in `spmf_pcb/initialization.py`. Repeat the paired command for seeds 3–16 to run all fourteen pairs.

Add `--dry-run` to check initialization, RGB–HSI routing, and one loss forward without optimizer updates. Supplied checkpoints and records are from completed runs. Complete retraining through this packaged entry point has not been repeated; bitwise agreement on different hardware/software is not guaranteed. The main initial state was recovered from the recorded training path; the paired initial states were saved separately.

## Forward throughput

```bash
python reproduce.py benchmark --output outputs/efficiency
```

The architecture-level benchmark uses the supplied frozen/joint timing checkpoints and prepared timing inputs, FP16, batch 1, 640×640, Conv/BN fusion, and alternating synchronized measurements. Recorded throughput is 52.859 and 52.554 FPS, respectively. These are separate architecture-level measurements, not measurements of the three main accuracy checkpoints. Preprocessing, transfer, and NMS are excluded; a new run reports its actual measured times.

## Repository layout

```text
reproduce.py          Command-line entry point
spmf_pcb/             Preprocessing, SPMF interface, training and evaluation
configs/              Recipes, board split, annotations and registration parameters
transforms/           Fitted LDA transforms and normalization parameters
reference/            Reference metrics and input checksums
environments/         Dependency versions
third_party/          Required Ultralytics source snapshots
weights/              Installed from release assets; ignored by Git
results/              Installed from release assets; ignored by Git
data/                 Generated locally from the official download; ignored by Git
outputs/              New experiment outputs; ignored by Git
```

The public interface is `spmf_pcb.model.SPMF`. Serialized checkpoint class identifiers are retained in the compatibility implementation so the original weights load without changing tensor keys or operations.

## License and dataset attribution

Code is distributed under **AGPL-3.0**, with the original third-party notices retained. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). PCB-Vision is distributed by its authors under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The supplied detection annotations, split, registration parameters, and preprocessing describe the adaptations used here; the dataset authors do not endorse this repository.

Please cite the original dataset when using it:

```bibtex
@dataset{arbash2024pcbvision,
  author = {Arbash, Elias and Fuchs, Margret and Rasti, Behnood and
            Lorenz, Sandra and Ghamisi, Pedram and Gloaguen, Richard},
  title = {PCB-Vision: A Multiscene RGB-Hyperspectral Benchmark Dataset of Printed Circuit Boards},
  year = {2024},
  version = {1},
  publisher = {RODARE},
  doi = {10.14278/rodare.2704}
}
```
