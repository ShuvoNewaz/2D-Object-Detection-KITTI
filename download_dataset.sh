#!/bin/bash
datadir=dataset
kitti_home_page=https://s3.eu-central-1.amazonaws.com/avg-kitti
training_labels=data_object_label_2.zip
left_images=data_object_image_2.zip

mkdir -p "${datadir}"

declare -a items_to_download=("${training_labels}"
                                "${left_images}"
                            )
for item in "${items_to_download[@]}";
do
    wget "${kitti_home_page}/${item}"
    unzip "${item}" -d "${datadir}"
    rm "${item}"
done