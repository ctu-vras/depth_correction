# Depth Correction

![](./docs/imgs/depth_correction_scheme.png)

LiDAR measurements correction models trained in a self-supervised manner on real data from diverse environments.
Models exploit multiple point cloud measurements of the same scene from different view-points in
order to reduce the bias based on the consistency of the constructed map.

- **self-supervised training** pipeline based on the point cloud map consistency information.

- depth correction models **remove LiDAR measurements bias** related to measuring
scene surfaces with high incidence angle.

- more **accurate maps** created from corrected consistent measurements.

- **reduction of the localization drift** in SLAM scenarios.


## Datasets

For models evauation we utilize and provide training pipeline on the following datasets:

- [ASL laser](https://projects.asl.ethz.ch/datasets/doku.php?id=laserregistration:laserregistration),
- [Semantic KITTI](http://www.semantic-kitti.org/dataset.html).


## Installation

Please, follow the installation instructions, provided in
[docs/install.md](https://github.com/RuslanAgishev/depth_correction/blob/main/docs/install.md)
