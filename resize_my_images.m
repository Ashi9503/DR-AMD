clear all;close all;clc;
srcFiles = dir('Y:\OTHER WORKS\2024 - 2025\BUSINESS PROJECTS\NOVEMBER - 2024\TK166370 - AMD diagnosis using OCT images and Diabetic diagnosis using Fundus Images in Matlab\CODE\AMD DATASET Stages\Wet Age-related Macular Degeneration\*.JPG');  % the folder in which ur images exists
for i = 1 : length(srcFiles)
filename = strcat('Y:\OTHER WORKS\2024 - 2025\BUSINESS PROJECTS\NOVEMBER - 2024\TK166370 - AMD diagnosis using OCT images and Diabetic diagnosis using Fundus Images in Matlab\CODE\AMD DATASET Stages\Wet Age-related Macular Degeneration\',srcFiles(i).name);
img = imread(filename);


%% Check if the image is RGB and convert to Grayscale if needed
if size(img, 3) == 3 % Check if the image has 3 channels (RGB)
    gray_img = rgb2gray(img); % Convert the image to grayscale
else
    gray_img = img; % Image is already grayscale
end

%% Resize the Grayscale Image
Resize_img = imresize(gray_img, [224 224]); 
figure; 
imshow(Resize_img); 
title('Resized Grayscale Image');

newfilename=fullfile('Y:\OTHER WORKS\2024 - 2025\BUSINESS PROJECTS\NOVEMBER - 2024\TK166370 - AMD diagnosis using OCT images and Diabetic diagnosis using Fundus Images in Matlab\CODE\AMD Resized Stages\Wet Age-related Macular Degeneration\',srcFiles(i).name);
imwrite(Resize_img,newfilename,'png');
end

close all