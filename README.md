# Hybrid Computer Vision Pipeline

## Objective

The objective of this assignment was to combine traditional image processing and artificial intelligence in one computer vision pipeline.

## 1. Image Loading

I used the peppers sample image. The image contains different coloured peppers and a background.

## 2. Traditional Image Segmentation

I converted the RGB image into Lab colour space and used K-means clustering to divide the image into two colour groups.

I created a binary mask to separate the foreground from the background. This helped prepare the image for AI classification.

## 3. AI Classification

I used the pre-trained SqueezeNet CNN for image classification.

SqueezeNet classified the original image and the segmented image. I recorded the predicted classes and confidence values.

## 4. Results and Comparison

I compared the classification results before and after traditional segmentation.

The confidence values show how certain the network is about its predictions. A higher confidence does not always mean that the classification is correct.

The results depend on the quality of segmentation and the content of the image.

## 5. Conclusion

This experiment demonstrated how traditional image processing and AI can work together.

K-means segmentation prepares the image by separating colour regions. SqueezeNet then classifies the prepared image.

One limitation is that K-means does not always separate the complete object correctly. Incorrect segmentation can affect the AI classification result.

## AI Usage

**Tool used:** ChatGPT

**How I used it:** I used ChatGPT to understand K-means segmentation, Lab colour space and SqueezeNet classification.

**What I verified or changed:** I ran the MATLAB script, checked the segmentation mask and compared the original and segmented image classification results.
