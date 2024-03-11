# Aligning-Sequoia-Images


## Description
This repository contains Python code for aligning Near-Infrared (NIR) and Red (RED) images captured by Sequoia Camera to calculate the Normalized Difference Vegetation Index (NDVI). The code utilizes the AlignmentSequoia class, which implements feature matching and homography estimation techniques to align the images accurately. The NDVI is computed both before and after the alignment process, allowing for the comparison of results. Depending on the specific characteristics of the images, it may be necessary to tune the parameters of the alignment process. To assist with this, the repository provides a function called generate_alignments, which helps find the parameters that produce the lowest error.

## Key Features
- Python code for image alignment using feature matching techniques.
- Calculation of NDVI before and after alignment.
- Visualizations of NDVI results for comparison.
- Easily customizable parameters for alignment and NDVI calculation.
- Function to generate alignments and fine-tune parameters automatically.

## Dependencies version  ( Python 3.11.3 )
- OpenCV (cv2) : 4.9.0
- NumPy : 1.26.4
- Matplotlib: 3.7.1

## Usage
1. Load NIR and RED images captured by Sequoia Camera.
2. Instantiate the AlignmentSequoia class and specify parameters if needed.
3. Call the alignment method to align the images.
4. Calculate NDVI before and after alignment.
5. Visualize the NDVI results for comparison.
6. Use the generate_alignments function to fine-tune parameters for optimal alignment.

## NDVI Comparison
Below is an example of the NDVI image before and after alignment:

![beforeandafter](https://github.com/KeonyJR/Aligning-Sequoia-Images/assets/10182525/a47ff90d-1ad5-46ea-9dc9-cc445743cfd6)

