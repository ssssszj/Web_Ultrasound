import os
import pandas as pd
import argparse


def annotation_maker(annotation_file: str = None, path_to_images: str = None, file_to_save: str = None):
    """

    Args:
        annotation_file:
        path_to_images:
        file_to_save:

    Returns:

    """
    annotation = pd.read_csv(annotation_file, ",", names=["filename", "label"], usecols=[0, 5])
    filename = [f for f in os.listdir(path_to_images) if "Bezier" in f]

    files = []
    for file in filename:
        files.append(file)

    data = pd.DataFrame(files, columns=["filename"])
    data["file"] = data["filename"].str.split("_", expand=True)[2]
    final = pd.merge(annotation, data, left_on="filename", right_on="file", how="outer")
    final = final.dropna()
    final = final[["filename_y", "label"]].rename({"filename_y": "filename"}, axis="columns")
    final = final.append(annotation)
    final.to_csv(file_to_save, sep=",", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_file", type=str, default="../../data/annotations.csv", help="Path to annotations file")
    parser.add_argument("--path_to_images", type=str, default="../../data/Bezier", help="Path to directory with images")
    parser.add_argument("--saved_filename", type=str, default="../../data/annotations_final.csv", help="Saved filename with *csv")
    opt = parser.parse_args()
    annotation_maker(opt.annotation_file, opt.path_to_images, opt.saved_filename)