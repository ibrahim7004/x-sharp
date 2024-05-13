"""Runs a pre-trained classifier on a set of images and reports accuracy."""
# ===================================================================

# Example : apply a specific pre-trained classifier to the  images
# path to the directory containing images is specified on the command line
# e.g. python classify.py --data=path_to_data
# path to the pre-trained network weights is given on the command line
# e.g. python classify.py --model=path_to_model
# python classify.py --data=xray_images --model=classifier.model

# Author : Amir Atapour Abarghouei, amir.atapour-abarghouei@durham.ac.uk

# Copyright (c) 2023 Amir Atapour Abarghouei

# License : LGPL - http://www.gnu.org/licenses/lgpl.html

# ===================================================================

import os
import argparse
import cv2

# ===================================================================

# parse command line arguments for paths to the data and model

parser = argparse.ArgumentParser(
    description='Perform image classification on x-ray images!')

parser.add_argument(
    "--data",
    type=str,
    help="specify path to the images",
    default='xray_images')

parser.add_argument(
    "--model",
    type=str,
    help="specify path to model weights",
    default='classifier.model')

args = parser.parse_args()

# ===================================================================

# load model weights:

model =  cv2.dnn.readNetFromONNX(args.model)

# lists to keep filenames, images and identifiers for healthy and sick labels:

names = []
images = []
healthys = []
pneumonias = []

# the first 50 images are healthy and the next 50 are not:

for i in range(1, 51):
    healthys.append(f'im{str(i).zfill(3)}')

for i in range(51, 101):
    pneumonias.append(f'im{str(i).zfill(3)}')

# read all the images from the directory

for file in os.listdir(args.data):
    names.append(file)
names.sort()

# remove any extra files Mac might have put in there:

if ".DS_Store" in names:
    names.remove(".DS_Store")

# keeping track of the number of correct predictions for accuracy:
correct = 0

# main loop:
for filename in names:

    # read image:
    img = cv2.imread(os.path.join(args.data, filename))

    if img is not None:

        # pass the image through the neural network:
        blob = cv2.dnn.blobFromImage(img, 1.0 / 255, (256, 256),(0, 0, 0), swapRB=True, crop=False)
        model.setInput(blob)
        output = model.forward()

        # identify what the predicted label is:
        if(output > 0.5):
            print(f'{filename}: pneumonia')
            if(filename.startswith(tuple(pneumonias))):
                correct += 1
        else:
            print(f'{filename}: healthy')
            if(filename.startswith(tuple(healthys))):
                correct += 1

# print final accuracy:
print(f'Accuracy is {correct/len(names)}')

# ===================================================================
