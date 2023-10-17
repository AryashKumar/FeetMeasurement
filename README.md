# FeetMeasurement
This Python code is designed to process an image to extract information related to the size of a person's feet in centimeters. It includes various image processing steps, such as preprocessing, clustering, edge detection, and bounding box extraction.

# Code Structure

The code is organized into functions, each serving a specific purpose. Here's a brief overview of the functions:
1. preprocess(img): This function performs image preprocessing, converting the image to HSV color space, applying a Gaussian blur, and normalizing the pixel values.

2. plotImage(img): This function displays an image using matplotlib.

3. cropOrig(bRect, oimg): Given a bounding rectangle bRect and an original image oimg, this function crops the region of interest from the original image.

4. overlayImage(croppedImg, pcropedImg): This function overlays a cropped image on top of a blank canvas, creating a visual representation of the extracted region of interest.

5. kMeans_cluster(img): Performs K-means clustering on the input image to segment it into two clusters.

6. getBoundingBox(img): Extracts the bounding boxes of the contours in the image and returns information about these contours.

7. drawCnt(bRect, contours, cntPoly, img): Draws the contours and bounding rectangle on a blank canvas using random colors.

8. edgeDetection(clusteredImage): Performs edge detection on the clustered image.

9. The main part of the code loads an image, applies the above functions to process and analyze it, and calculates the size of the feet.

# Usage
To use this code, you can follow these steps:

1. Replace 'barefeet8.jpg' in the oimg = imread('barefeet8.jpg') line with the path to the image you want to analyze.

2. Uncomment or comment out the plotImage() calls to view intermediate results if desired.

3. Run the script, and it will process the image and display the feet size in centimeters.

# Dependencies
To run this code, you need the following Python libraries installed:

1. skimage
2. numpy
3. matplotlib
4. scipy
5. imutils
6. OpenCV (cv2)
7. scikit-learn (sklearn)
