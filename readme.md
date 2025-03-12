# 2D Object Detection

## Dataset

The dataset used for this project is the [KITTI 2D Object Detection Evaluation](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d). The parts of this dataset used are:
1. Left Color Images (12 GB)
2. Training Labels (5 MB)

Enter `bash download_dataset.sh` in your terminal to automatically download and organize the dataset. The dataset will require about 12 GB of free space and, depending on the internet speed, may take a while to download.

### Details of the Dataset

#### Images

The RGB images depict a variety of scenes containing different objects. There are 7481 images in the training set and 7518 in the validation/test set.

#### Labels

The lables exist only for the training set. These `.txt` files contain the following:

1. The object class.
2. Truncation - A measure of how much the object out of image bounds. 0 $\rightarrow$ non-truncated. 1 $\rightarrow$ truncated.
3. Occlusion. 0 $\rightarrow$ not occlude. 1 $\rightarrow$ partly occluded. 2 $\rightarrow$ mostly occluded. 3 $\rightarrow$ unknown.
4. Observation angle $[-\pi,\pi]$.
5. 2D bounding box of objects in the image. Contains coordinates of 4 corners.
6. 3D object dimensions in meters.

    i. Height
    
    ii. Width
    
    iii. length
7. 3D center location in camera coordinates.
8. Rotation angle around the y-axis $[-\pi,\pi]$.

For the 2D object detection task, we only need the object class and the bounding box coordinates.

## TASKS COMPLETED

1. Dataloader with collate_fn for consistent image dimensions.
2. Model structure.
3. Metrics.
4. Trainer

## TASKS REMAINING AND ISSUES

1. Training MAP not increasing.
2. GPU runs out of memory.
3. Validation
