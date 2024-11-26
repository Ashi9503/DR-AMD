%% Load Image
[file, path] = uigetfile('C:\Users\Rakshit Preran B\Downloads\TK166370\TK166370 - AMD diagnosis using OCT images and Diabetic diagnosis using Fundus Images in Matlab\CODE\TESTDATA_AMD\.', 'Select an image');
img = imread([path, file]);  


%% Check if the image is RGB and convert to Grayscale if needed
if size(img, 3) == 3 % Check if the image has 3 channels (RGB)
    gray_img = rgb2gray(img); % Convert the image to grayscale
else
    gray_img = img; % Image is already grayscale
end

%% Resize the Grayscale Image
Resize_img = imresize(gray_img, [224 224]); 


% Step 3: Enhance the contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
enhancedImage = adapthisteq(gray_img);

% Step 4: Apply edge detection to segment the regions of interest
edges = edge(enhancedImage, 'Canny');

% Step 5: Morphological operations to clean up the segmentation
se = strel('disk', 3);  % Structuring element
dilatedEdges = imdilate(edges, se);  % Dilate to fill gaps
cleanedEdges = imerode(dilatedEdges, se);  % Erode to remove noise

% Step 6: Apply a median filter to smooth the segmented image
filteredImage = medfilt2(cleanedEdges, [3 3]);

% Step 7: Display the Enhanced, Segmented, and Filtered Images
figure;

subplot(2, 3, 1);
imshow(img);
title('Original Image');

subplot(2, 3, 2);
imshow(gray_img);
title('Grayscale Image');

subplot(2, 3, 3);
imshow(Resize_img);
title('Resized Image');

subplot(2, 3, 4);
imshow(cleanedEdges);
title('Segmented Image (Edges)');

subplot(2, 3, 5);
imshow(filteredImage);
title('Filtered Image');

% Save results
imwrite(cleanedEdges, 'segmented_image.jpg'); % Segmented Image
imwrite(filteredImage, 'filtered_image.jpg'); % Filtered Image

%% Prepare Data for Training
matlabroot = pwd;
datasetpath = fullfile(matlabroot, 'AMD Resized_Images');
imds = imageDatastore(datasetpath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');

count = imds.countEachLabel;

augimdsTrain = augmentedImageDatastore([224 224 1], imdsTrain);
augimdsValidation = augmentedImageDatastore([224 224 1], imdsValidation);

net = resnet50;

layers = [
    imageInputLayer([224 224 1])
    net(2:end-2) % Accessing the pretrained network layers from second layer to end-3 layers
    fullyConnectedLayer(2) % Modifying the fully connected layer with respect to classes
    softmaxLayer
    classificationLayer
];

opts = trainingOptions("sgdm", ...
    "ExecutionEnvironment", "auto", ...
    "InitialLearnRate", 0.001, ...
    "MaxEpochs", 100, ...
    "MiniBatchSize", 25, ...
    "Shuffle", "every-epoch", ...
    "ValidationData", augimdsValidation, ...
    "ValidationFrequency", 50, ...
    "Verbose", true, ... 
    "Plots", "training-progress");

% [AMD_training, AMD_traininginfo] = trainNetwork(augimdsTrain, layers, opts); 

load AMD_training
load AMD_traininginfo

% Classify the resized grayscale image
YPred = classify(AMD_training, Resize_img);
msgbox(char(YPred))
fprintf('The training accuracy by ResNet50 Net is %0.4f\n', mean(AMD_traininginfo.TrainingAccuracy));

[YPred_data, scores] = classify(AMD_training, augimdsValidation);
YValidation = imdsValidation.Labels;

% Calculate confusion matrix
[m, order] = confusionmat(YValidation, YPred_data);
figure;
cm = confusionchart(m, order);

%%
output = char(YPred);

if strcmp(output, 'Age-related Macular Degeneration')
    
     Diseasematlabroot = cd;    % Dataset path
     Diseasedatasetpath = fullfile(Diseasematlabroot,'AMD Resized Stages');   %Build full file name from parts
     Diseaseimds = imageDatastore(Diseasedatasetpath,'IncludeSubfolders',true,'LabelSource','foldernames');    %Datastore for image data

     [DiseaseimdsTrain, DiseaseimdsValidation] = splitEachLabel(Diseaseimds,0.8);     %Split ImageDatastore labels by proportions

     DiseaseaugimdsTrain = augmentedImageDatastore([224 224 1],DiseaseimdsTrain);  %Generate batches of augmented image data
     DiseaseaugimdsValidation = augmentedImageDatastore([224 224 1],DiseaseimdsValidation);
        % Training Options
     
     Diseaseoptions = trainingOptions("sgdm",...
          "ExecutionEnvironment","auto",...
          "InitialLearnRate",0.001,...
          "MaxEpochs",100,...
          "MiniBatchSize",25,...
          "Shuffle","every-epoch",...
          "ValidationData",DiseaseaugimdsTrain, ...
          "ValidationFrequency",50,...
          "Verbose",true, ... 
          "Plots","training-progress");
      
     Resnet_net = resnet50;
       
     Diseaselayers = [imageInputLayer([224 224 1]) 
           
             Resnet_net(2:end-2) %accessing the pretrained network layers from second layer to end-3 layers
               
             fullyConnectedLayer(3) % modifying the fullyconnected layer with respect to classes
               
             softmaxLayer
               
             classificationLayer];
           
%     [AMD_Diseasenet, AMD_Diseasetraininfo] = trainNetwork(DiseaseaugimdsTrain,Diseaselayers,Diseaseoptions);  %Train neural network for deep learning
         
      load AMD_Diseasenet
      load AMD_Diseasetraininfo

       [DiseaseYPred,Diseasescore] = classify(AMD_Diseasenet,Resize_img);      %Classify data using a trained deep learning neural network
       msgbox(char(DiseaseYPred));
       fprintf('The Stage training accuracy by resnet50 Net is %0.4f\n', mean(AMD_Diseasetraininfo.TrainingAccuracy));

       
       [DiseaseYPred_data, scores] = classify(AMD_Diseasenet, DiseaseaugimdsValidation);
       DiseaseYValidation = DiseaseimdsValidation.Labels;
       
       % Calculate confusion matrix
       [Diseasem,Diseaseorder] = confusionmat(DiseaseYValidation, DiseaseYPred_data);
       figure
       cm = confusionchart(Diseasem,Diseaseorder);
end
end