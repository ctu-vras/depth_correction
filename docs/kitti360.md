# KITTI-360

Instructions how to run the `depth_correction` package with the
[KITTI-360](https://www.cvlibs.net/datasets/kitti-360/) dataset.
Prerequisite: build the `depth_correction` package (follow [install.md](./install.md)).

## Prepare the data

1. Download the point clouds, corresponding poses and calibration data from the dataset.
Save it in the `depth_correction/data/` folder or configure the `python` path as proposed in
[kitti360Scripts](https://github.com/autonomousvision/kitti360Scripts.git) repository:

    ```bash
    export KITTI360_DATASET=/PATH/TO/THE/DATASET
    ```
    The downloaded dataset should have the following structure:
    ```commandline
    kitti360
    ├── calibration
    ├── data_3d_raw
    │   ├── 2013_05_28_drive_0000_sync
    │   ├── 2013_05_28_drive_0002_sync
    │   ├── 2013_05_28_drive_0003_sync
    │   ├── 2013_05_28_drive_0004_sync
    │   ├── 2013_05_28_drive_0005_sync
    │   ├── 2013_05_28_drive_0006_sync
    │   ├── 2013_05_28_drive_0007_sync
    │   ├── 2013_05_28_drive_0009_sync
    │   └── 2013_05_28_drive_0010_sync
    ├── data_3d_semantics
    │   └── train
    └──data_poses
        ├── 2013_05_28_drive_0000_sync
        ├── 2013_05_28_drive_0002_sync
        ├── 2013_05_28_drive_0003_sync
        ├── 2013_05_28_drive_0004_sync
        ├── 2013_05_28_drive_0005_sync
        ├── 2013_05_28_drive_0006_sync
        ├── 2013_05_28_drive_0007_sync
        ├── 2013_05_28_drive_0008_sync
        ├── 2013_05_28_drive_0009_sync
        ├── 2013_05_28_drive_0010_sync
        └── 2013_05_28_drive_0018_sync
    ```
   The `data_3d_semantics` data is utilized for the step 3.

2. Install `kitti360Scripts` with `pip`:
   ```
   pip install git+https://github.com/autonomousvision/kitti360Scripts.git
   ```
   We use open3D to visualize 3D point clouds:
   ```
   pip install open3d
   ```
   
3. (Optional) Generate point cloud scans without dynamic objects:
   ```commandline
   roscd depth_correction/scripts/
   ./generate_scans_wo_dynamic_objects_kitti360
   ```
   In this way the following folder should be generated which exhibits the same structure as `data_3d_raw`:
   ```commandline
   kitti360
    └──data_3d_filtered
   ```
   
4. (Optional) To inspect the downloaded data run:
   ```commandline
   python -m depth_correction.datasets.kitti360
   ```
   
## Evaluate SLAM on KITTI-360

In order to evaluate localization accuracy using []() SLAM pipeline with the sequences from KITTI-360 run:

```bash
roslaunch depth_correction slam_eval.launch dataset:=kitti360/03_start_2_end_200_step_1
```
Note that the sequence name from the dataset is given by the following format: `NN_start_SS_end_EE_step_ss`,
where `NN` is sequence number, `SS` and `EE` are the first and the last scan numbers respectively,
`ss` denotes the data step.

To run the SLAM algorithm with the depth correction method:

```bash
roslaunch depth_correction slam_eval.launch model_class:=ScaledPolynomial model_kwargs:="{'exponent': [4], 'w': [-0.001]}" depth_correction:=true dataset:=kitti360/03_start_2_end_200_step_1
```

As a result you should see the rotation and translation errors computed
with respect to the ground truth data from KITTI-360.
