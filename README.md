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

## Checkpoints and reference predictions

Download `PCB-SPMF-assets-v1.0.0.zip` from this repository's **Releases → Assets** and extract it into the repository root:

```bash
python -m zipfile -e /path/to/PCB-SPMF-assets-v1.0.0.zip .
python reproduce.py verify
```

The asset file contains trained weights, unadapted initialization states, reference predictions, training curves, and bootstrap samples. It contains no PCB images or hyperspectral cubes. Before installing assets, `python reproduce.py verify --code-only` checks the source distribution alone.

## Evaluation

Run in the CUDA environment after preparing the data and installing the assets:

```bash
python reproduce.py evaluate --profile main --output outputs/main
python reproduce.py evaluate --profile paired --batch 16 --output outputs/paired_primary
python reproduce.py evaluate --profile paired --batch 8 --output outputs/paired_sensitivity
```

The evaluator runs RGB, SPMF, and SPMF-disabled inference, checks the input branches, and compares predictions with the supplied references. A single paired seed can be selected with `--seeds 3`. Use a new output directory for each run.

| Experiment | Spectral fitting population | RGB mAP50:95 | SPMF mAP50:95 |
|---|---|---:|---:|
| Selected main checkpoint | All 53 boards | 0.635033 | 0.636350 |
| Fourteen paired initializations, batch 16 | Training 42 boards | 0.616832 | 0.619330 |

The main SPMF checkpoint has P = 0.904516, R = 0.919212, and mAP50 = 0.929834. Its matched RGB baseline is the same detector recovered by disabling SPMF. The main result is a selected checkpoint and uses all-board supervised spectral fitting; it is not a training-only preprocessing evaluation.

The paired experiment uses seeds 3–16, each with a separate RGB prior. Under the primary setting, the mean mAP50:95 increase is **0.249849 percentage points**, with 14 positive pairs. The seed 95% interval is [0.142963, 0.356736] and the PCB interval is [0.096770, 0.334835] percentage points. Batch-8 evaluation of the same models gives a mean increase of 0.111793 percentage points with 10 positive and 4 negative pairs. Both settings are retained. The 11 validation boards also supply checkpoint selection.

Recompute statistics from the supplied per-board predictions:

```bash
python reproduce.py replay-statistics --output outputs/statistics
python reproduce.py replay-statistics --bootstrap --output outputs/statistics_bootstrap
```

The second command recomputes all 2,000 shared PCB bootstrap resamples. Without `--bootstrap`, point estimates and seed intervals are recomputed while PCB intervals use the supplied resampling array. These commands do not perform network inference.

## Training

```bash
python reproduce.py train-rgb --seed 3 --output outputs/rgb_seed_03
python reproduce.py train-frozen --profile main --seed 0 --output outputs/main_adaptation
python reproduce.py train-frozen --profile paired --seed 3 --output outputs/spmf_seed_03
```

RGB pretraining uses 500 epochs. Frozen adaptation uses 120 epochs and updates only the HSI branch and SPMF while preserving the RGB parameters and batch-normalization state. `train-frozen` starts from the corresponding supplied unadapted state; it does not automatically consume the output of a preceding `train-rgb` command. Tensor migration for a new RGB prior is implemented in `spmf_pcb/initialization.py`.

Add `--dry-run` to `train-frozen` to check initialization, paired inputs, and a loss forward pass without optimizer updates. Fixed-checkpoint evaluation and preprocessing were checked in the reference environments; a complete rerun of every 500+120 epoch training sequence through this packaged entry point has not been performed. Exact retraining across different hardware/software is not guaranteed. The main initialization is recovered from the original recorded path; a contemporaneous immutable hash was not available. The paired initialization states were saved individually.

## Forward throughput

```bash
python reproduce.py benchmark --output outputs/efficiency
```

This uses the separate prepared timing inputs and the fixed frozen/joint checkpoints, FP16, batch 1, 640×640, Conv/BN fusion, and alternating synchronized measurements. It measures network forward time, excluding preprocessing, transfer, and NMS. The recorded throughputs are 52.859 and 52.554 FPS; a new run reports its actual measured times.

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
