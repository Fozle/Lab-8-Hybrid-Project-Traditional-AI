
%% Hybrid Computer Vision: Traditional Segmentation + AI Classification
% K-means Color Segmentation and SqueezeNet CNN

close all;
clear;
clc;

%% 1. Load image

targetImage = imread('peppers.png');

% Ensure the image has three color channels
if size(targetImage,3) ~= 3
    error('Please use an RGB color image.');
end

fprintf('Original image loaded successfully.\n');

%% 2. Load pre-trained SqueezeNet network

disp('Loading pre-trained AI network (SqueezeNet)...');

net = squeezenet;

disp('SqueezeNet loaded successfully.');

inputSize = net.Layers(1).InputSize;

%% 3. Traditional method: K-means color segmentation

disp('Applying K-means color segmentation...');

% Convert RGB image to Lab color space
imgLab = rgb2lab(targetImage);

% Extract a and b color channels
ab = im2single(imgLab(:,:,2:3));

% Divide pixels into two color clusters
pixelLabels = imsegkmeans(ab,2,'NumAttempts',3);

% Select the brighter cluster as the foreground
L = imgLab(:,:,1);

meanL = zeros(2,1);

for k = 1:2
    meanL(k) = mean(L(pixelLabels == k));
end

[~, objectCluster] = max(meanL);

mask = pixelLabels == objectCluster;

% Clean small segmentation noise
mask = imclose(mask,strel('disk',3));
mask = imfill(mask,'holes');

% Isolate the object
isolatedObject = targetImage;
isolatedObject(repmat(~mask,1,1,3)) = 0;

%% 4. Prepare images for AI classification

originalInput = imresize(targetImage, ...
    [inputSize(1) inputSize(2)]);

resizedObject = imresize(isolatedObject, ...
    [inputSize(1) inputSize(2)]);

%% 5. AI classification: original image

disp('Classifying original image...');

[originalPrediction,originalScores] = ...
    classify(net,originalInput);

originalConfidence = max(originalScores) * 100;

%% 6. AI classification: segmented image

disp('Classifying segmented image...');

[objectPrediction,objectScores] = ...
    classify(net,resizedObject);

objectConfidence = max(objectScores) * 100;

%% 7. Display traditional segmentation results

figure(1);

subplot(2,2,1);
imshow(targetImage);
title('Original Image');

subplot(2,2,2);
imshow(mask);
title('K-means Segmentation Mask');

subplot(2,2,3);
imshow(isolatedObject);
title('Segmented Image');

subplot(2,2,4);
imshow(resizedObject);
title('Resized AI Input');

sgtitle('Traditional Image Segmentation');

saveas(gcf,'Lab08_segmentation.png');

%% 8. Compare AI classification results

figure(2);

subplot(1,2,1);
imshow(originalInput);
title(sprintf('Original: %s (%.2f%%)', ...
    string(originalPrediction),originalConfidence));

subplot(1,2,2);
imshow(resizedObject);
title(sprintf('Segmented: %s (%.2f%%)', ...
    string(objectPrediction),objectConfidence));

sgtitle('SqueezeNet Classification Comparison');

saveas(gcf,'Lab08_classification.png');

%% 9. Print results

fprintf('\nHYBRID PIPELINE RESULTS\n');
fprintf('------------------------------------\n');

fprintf('Original image classification: %s\n', ...
    string(originalPrediction));

fprintf('Original confidence: %.2f%%\n', ...
    originalConfidence);

fprintf('Segmented image classification: %s\n', ...
    string(objectPrediction));

fprintf('Segmented confidence: %.2f%%\n', ...
    objectConfidence);

fprintf('\nAll figures saved successfully.\n');
fprintf('Hybrid pipeline completed.\n');
