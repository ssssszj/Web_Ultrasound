"""Data augmentation for Cervix Dataset"""

# Import necessary packages
import Augmentor
import glob
import os
import shutil
import argparse


def data_augmentation(path_to_images, path_to_masks):
    """

    Args:
        path_to_images: path to directory with images
        path_to_masks: path to directory with masks

    Returns:
        Augmented data.
    """
    p = Augmentor.Pipeline(path_to_images)
    p.ground_truth(path_to_masks)
    p.random_contrast(probability=1, min_factor=0.7, max_factor=1)
    p.random_brightness(probability=1, min_factor=0.95, max_factor=1.05)
    p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
    p.sample(3500)

    for image in glob.glob(path_to_images + '/output/*.png'):
        shutil.move(image, path_to_images)
    shutil.rmtree(path_to_images + "/output")
    for image in glob.glob(path_to_images + '/_groundtruth*.png'):
        shutil.move(image, path_to_masks)

    for img in os.listdir(path_to_masks):
        if '_groundtruth_' in img:
            dest_image = img.replace('_groundtruth_(1)_Bezier', 'Bezier_origin')
            os.rename(os.path.join(path_to_masks, img), os.path.join(path_to_masks, dest_image))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_images", type=str, default="../../data/Bezier", help="Path to images directory")
    parser.add_argument("--path_to_mask", type=str, default="../../data/Beziermask", help="Path to masks directory")
    opt = parser.parse_args()
    data_augmentation(opt.path_to_images, opt.path_to_mask)