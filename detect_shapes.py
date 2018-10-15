#! /usr/bin/env python

import argparse
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt


def detect_boxes(input_dir, output_dir):
    """
    The function detects the rectangle boxes in a given image.
    :param input_dir: The input file to images
    :param output_dir: The output files to images
    :return: None, output files will be written in given
    """

    if os.path.exists(input_dir):
        for images in os.listdir(input_dir):
            file = os.path.join(input_dir, images)
            image = cv2.imread(file)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # blur, to remove sharp edges
            blurred_image = cv2.GaussianBlur(gray, (7, 7), 0)
            # adaptively thresholding to make binary
            thresholded = cv2.adaptiveThreshold(
                blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 3)

            edged = cv2.Canny(thresholded, 30, 200)
            # find countours in the image
            _, contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            rects = []
            for items in contours:
                # approximating the contour
                peri = cv2.arcLength(items, True)
                if peri >20:
                    perimeter = 0.01 * peri
                    approx = cv2.approxPolyDP(items, perimeter, True)
                    # check if the approximated contour has four points
                    if len(approx) == 4:
                        rects.append(approx)

            # drawing the countours
            image_copy = image.copy()
            countered = cv2.drawContours(image_copy, rects, -1, (0, 0, 255), 2)
            if os.path.exists(output_dir):
                cv2.imwrite(os.path.join(output_dir, images + "result.jpg"), countered)
            else:
                os.makedirs(output_dir)
                cv2.imwrite(images + "result.jpg", countered)

    else:
        print("Input path does not exist")
        exit(1)


def detect_checkboxes(input_dir, output_dir):
    """
    The function detects all the rectangles in an image using template matching.
    A better solution is using SHIFT, However SHIFT is not available in the free opencv due to patent constraints.
    :param input_dir: The input file to images
    :param output_dir: The output files where images are stored
    :return:None, Output files will be written in a given output dir
    """
    template_file = "template"
    if os.path.exists(input_dir):
        for image in os.listdir(input_dir):
            file = os.path.join(input_dir, image)
            img_rgb = cv2.imread(file)
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
            if os.path.exists(template_file):
                for templates in os.listdir(template_file):
                    template_path = os.path.join(template_file, templates)
                    template = cv2.imread(template_path, 0)
                    w, h = template.shape[::-1]
                    # matching the template
                    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
                    threshold = 0.7
                    loc = np.where(res >= threshold)
                    count = 0
                    max_count = 0
                    for pt in zip(*loc[::-1]):
                        x = pt[0] + w
                        y = pt[1] + h
                        cv2.rectangle(img_rgb, pt, (x, y), (0, 0, 255), 2)
                        count += 1
                        if count > max_count:
                            max_count = count
                            if os.path.exists(output_dir):
                                cv2.imwrite(os.path.join(output_dir, image + "result.jpg"), img_rgb)
                            else:
                                os.makedirs(output_dir)
                                cv2.imwrite(image + "result.jpg", img_rgb)
            else:
                print("Provide proper template path")
                exit(1)
    else:

        print("Input path does not exist")
        exit(1)

    exit(0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lets detect boxes')
    parser.add_argument('input_directory', help='Location of input images')
    parser.add_argument('output_directory', help='Location of output images')
    parser.add_argument('--lines', help='Pass this to detect lines')
    parser.add_argument('--boxes', help='Pass this to detect boxes')
    parser.add_argument('--checkboxes', help='Pass this to detect checkboxes')
    args = parser.parse_args()
    if args.checkboxes:
        detect_checkboxes(input_dir=args.input_directory,
                          output_dir=args.output_directory)
    if args.boxes:
        detect_boxes(input_dir=args.input_directory,
                     output_dir=args.output_directory)

