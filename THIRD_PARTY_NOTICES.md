# Third-party notices

## Ultralytics

The source snapshots under `third_party/fusion/` and `third_party/rgb/` derive from Ultralytics. Their copyright and AGPL-3.0 notices are retained. The supplied license is in `third_party/ULTRALYTICS_LICENSE`; the repository's code is distributed under the same license in `LICENSE`.

The fusion snapshot includes the multimodal model, SPMF compatibility implementation and associated parsing/loading modifications. Checkpoint class identifiers remain for serialization compatibility. Pretrained helper weights `yolov8n.pt` and `yolo11n.pt` are included in the release assets, together with the experiment's trained models; retain the applicable Ultralytics notices when redistributing these assets.

Upstream: https://github.com/ultralytics/ultralytics

License information: https://www.ultralytics.com/license

## PCB-Vision

Dataset: *PCB-Vision: A Multiscene RGB-Hyperspectral Benchmark Dataset of Printed Circuit Boards*, Version 1 (2024).

Authors: Elias Arbash, Margret Fuchs, Behnood Rasti, Sandra Lorenz, Pedram Ghamisi, and Richard Gloaguen.

Source: https://doi.org/10.14278/rodare.2704

License: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/

The raw images/cubes and processed spectral images are not redistributed in this release. Users obtain them from the official source and run the preprocessing commands. The supplied bounding-box annotations, registration parameters and board split accompany the transformations used in this work. The annotations and registration data derived from PCB-Vision retain CC BY 4.0 attribution; software licensing does not replace the dataset license. Changes include spectral selection/projection/normalization, image registration, the detection-box annotations and the train/validation organization. No endorsement by the dataset creators is implied.

The Ultralytics demonstration image under `third_party/fusion/ultralytics/assets/` is retained for the upstream training/AMP utilities; it is not a PCB dataset image.
