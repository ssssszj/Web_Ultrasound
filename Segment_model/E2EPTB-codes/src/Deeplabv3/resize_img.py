from PIL import Image
import glob
import os

path_img = "Resized_img"
if not os.path.exists(path_img):
    os.makedirs(path_img)

path_mask = "Resized_masks"
if not os.path.exists(path_mask):
    os.makedirs(path_mask)


for filename in glob.glob(f"../../data/*/*.png"):
    print(filename)
    if "Beziermask" in filename:
        mask = Image.open(filename).resize((512, 512))
        mask.save("{}{}{}".format(path_mask, "/", os.path.split(filename)[1]))
    elif "Bezier" in filename:
        img = Image.open(filename).resize((512, 512))
        img.save("{}{}{}".format(path_img, "/", os.path.split(filename)[1]))