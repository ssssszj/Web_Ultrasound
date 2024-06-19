"""Draw Bezier Curves on Cervix"""

# Import necessary packages

from PIL import ImageDraw
from PIL import Image
import json
import pandas as pd
from pandas.io.json import json_normalize
import csv
import glob
import cv2
import os


def extract_coordinates():
    annotations = pd.read_csv('../../data/annotations.csv', header=None, delimiter=',')
    array = annotations.values
    for item in array:
        image = item[0]
        annotation_json = json.loads(item[:][4])
        items = annotation_json['items']
        cervix = list(filter(lambda x: x['type'] == 'Cervix - closed', items))
        if cervix:
            cervix = cervix[0]
            points = [(float(cervix['x1']), float(cervix['y1'])), (float(cervix['x2']), float(cervix['y2'])),
                      (float(cervix['x3']), float(cervix['y3'])), (float(cervix['x4']), float(cervix['y4']))]
            start = points[0]
            control1 = points[1]
            control2 = points[2]
            end = points[3]

            yield (image, start, control1, control2, end)


def make_bezier(xys):
    """

    Args:
        xys:

    Returns:

    """
    # xys should be a sequence of 2-tuples (Bezier control points)
    n = len(xys)
    combinations = pascal_row(n - 1)

    def bezier(ts):
        """

        Args:
            ts:

        Returns:

        """
        # This uses the generalized formula for bezier curves
        # http://en.wikipedia.org/wiki/B%C3%A9zier_curve#Generalization
        result = []
        for t in ts:
            tpowers = (t ** i for i in range(n))
            upowers = reversed([(1 - t) ** i for i in range(n)])
            coefs = [c * a * b for c, a, b in zip(combinations, tpowers, upowers)]
            result.append(
                tuple(sum([coef * p for coef, p in zip(coefs, ps)]) for ps in zip(*xys)))
        return result

    return bezier


def pascal_row(n):
    """

    Args:
        n:

    Returns:

    """
    # This returns the nth row of Pascal's Triangle
    result = [1]
    x, numerator = 1, n
    for denominator in range(1, n // 2 + 1):
        x *= numerator
        x /= denominator
        result.append(x)
        numerator -= 1
    if n & 1 == 0:
        # n is even
        result.extend(reversed(result[:-1]))
    else:
        result.extend(reversed(result))
    return result


if __name__ == '__main__':

    inpaints_dir = '../../data/inpaintImages/'
    inpaint_suffix = '_inpaint.png'
    pathROI = '../../data/Bezier'

    if not os.path.exists(pathROI):
        os.mkdir(pathROI)

    radius = 30.0
    offset = 450

    for image, start, control1, control2, end in extract_coordinates():
        image_id = image.replace(".png", "")
        try:
            img_path = [file for file in glob.glob(inpaints_dir + image_id + "*.png")][0]
        except IndexError:
            continue
        im = Image.open(img_path)
        width, height = im.size
        draw = ImageDraw.Draw(im)
        ts = [t / width for t in range(width + 1)]

        xys = [start, control1, control2, end]
        bezier = make_bezier(xys)
        points = bezier(ts)
        for point in points:
            draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill='red')
        
        # im.crop((width//2 - offset//2, height//2 - offset//2, width//2 + offset//2, height//2 + offset//2)).save(os.path.join(pathROI, image))
        im.save(os.path.join(pathROI, image))
