#!/bin/bash

# Script to extract all images from the dataset
echo "Starting to extract ALL images from the dataset..."
echo "This may take a while as there are 10,783 training images and 1,542 test images"

python extract_images.py all

echo "All images extracted! Check the 'all_extracted_images' folder"
echo "Train images: all_extracted_images/train/"
echo "Test images: all_extracted_images/test/"
