function result = processImage(ImagePath);
%%
[file, path] = uigetfile('C:\Users\Rakshit Preran B\Downloads\TK166370\TK166370 - AMD diagnosis using OCT images and Diabetic diagnosis using Fundus Images in Matlab\CODE\TESTDATA_DR\.', 'Select an image');
img = imread([path, file]); 
figure; 
tiledlayout(2,4);
nexttile;
imshow(img);  
title('Input Image');

%%
Resize_img = imresize(img,[224 224]);       
nexttile
imshow(Resize_img); 
title('Resize Image');

if size(img, 3) == 3
    grayImage = rgb2gray(img);
else
    grayImage = img;
end
% Display grayscale image
nexttile
imshow(grayImage);
title('Grayscale Image');

% Step 2: Contrast enhancement
enhancedImage = imadjust(grayImage);

% Display enhanced image
nexttile
imshow(enhancedImage);
title('Contrast Enhanced Image');


% Step 3: Noise removal using median filtering
filteredImage = medfilt2(enhancedImage, [3, 3]);

% Display filtered image
nexttile
imshow(filteredImage);
title('Filtered Image');

% Step 4: Blood vessel segmentation using morphological operations
se = strel('disk', 1); % Structuring element
topHatFiltered = imtophat(filteredImage, se); % Top-hat filtering for background suppression

% Thresholding for segmentation
binaryImage = imbinarize(topHatFiltered, 'adaptive', 'Sensitivity', 0.4);

%Remove small objects and refine segmentation
cleanedImage = bwareaopen(binaryImage, 50); % Remove objects smaller than 50 pixels

% Step 6: Display segmentation result
nexttile
imshow(cleanedImage);
title('Segmented Image');

% Step 7: Overlay segmentation result on original image
segmentationOverlay = imoverlay(img, cleanedImage, [1 0 0]); % Red overlay

% Display final overlay result
nexttile
imshow(segmentationOverlay);
title('Segmentation Overlay on Original Image');

%%
matlabroot = pwd;
datasetpath = fullfile(matlabroot, 'DR Resized Images');
imds = imageDatastore(datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames');

[imdsTrain, imdsValidation] = splitEachLabel(imds,0.8,'randomized');

count = imds.countEachLabel;

augimdsTrain = augmentedImageDatastore([224 224 3],imdsTrain);
augimdsValidation = augmentedImageDatastore([224 224 3],imdsValidation);

net = inceptionv3;

layers = [imageInputLayer([224 224 3])
    
    net(2:end-2) %accessing the pretrained network layers from second layer to end-3 layers
    
    fullyConnectedLayer(2) % modifying the fullyconnected layer with respect to classes
    
    softmaxLayer
    
    classificationLayer];


opts = trainingOptions("sgdm",...
    "ExecutionEnvironment","auto",...
    "InitialLearnRate",0.001,...
    "MaxEpochs",100,...
    "MiniBatchSize",25,...
    "Shuffle","every-epoch",...
    "ValidationData",augimdsValidation, ...
    "ValidationFrequency",50,...
    "Verbose",true, ... 
    "Plots","training-progress");

% [DR_training, DR_traininginfo] = trainNetwork(augimdsTrain,layers,opts); 

load DR_training
load DR_traininginfo

YPred = classify(DR_training, Resize_img);
msgbox(char(YPred))
fprintf('The training accuracy by Inception V3 Net is %0.4f\n', mean(DR_traininginfo.TrainingAccuracy));

[YPred_data, scores] = classify(DR_training, augimdsValidation);
YValidation = imdsValidation.Labels;

% Calculate confusion matrix
[m,order] = confusionmat(YValidation, YPred_data);
figure
cm = confusionchart(m,order);

%%
output = char(YPred);

if strcmp(output, 'Diabetic Retinopathy')
     Diseasematlabroot = cd;    % Dataset path
     Diseasedatasetpath = fullfile(Diseasematlabroot,'DR Resized Images - Stages');   %Build full file name from parts
     Diseaseimds = imageDatastore(Diseasedatasetpath,'IncludeSubfolders',true,'LabelSource','foldernames');    %Datastore for image data

     [DiseaseimdsTrain, DiseaseimdsValidation] = splitEachLabel(Diseaseimds,0.8);     %Split ImageDatastore labels by proportions

     DiseaseaugimdsTrain = augmentedImageDatastore([224 224 3],DiseaseimdsTrain);  %Generate batches of augmented image data
     DiseaseaugimdsValidation = augmentedImageDatastore([224 224 3],DiseaseimdsValidation);
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
      
      InceptionV3_net = inceptionv3;
       
      Diseaselayers = [imageInputLayer([224 224 3]) 
           
               InceptionV3_net(2:end-2) %accessing the pretrained network layers from second layer to end-3 layers
               
               fullyConnectedLayer(4) % modifying the fullyconnected layer with respect to classes
               
               softmaxLayer
               
               classificationLayer];

%     [DR_Diseasenet, DR_Diseasetraininfo] = trainNetwork(DiseaseaugimdsTrain,Diseaselayers,Diseaseoptions);  %Train neural network for deep learning
         
      load DR_Diseasenet
      load DR_Diseasetraininfo
  
      [DiseaseYPred,Diseasescore] = classify(DR_Diseasenet,Resize_img);      %Classify data using a trained deep learning neural network
      msgbox(char(DiseaseYPred));
      fprintf('The Stage training accuracy by Inception V3 Net is %0.4f\n', mean(DR_Diseasetraininfo.TrainingAccuracy));

       
       [DiseaseYPred_data, scores] = classify(DR_Diseasenet, DiseaseaugimdsValidation);
       DiseaseYValidation = DiseaseimdsValidation.Labels;
       
       % Calculate confusion matrix
       [Diseasem,Diseaseorder] = confusionmat(DiseaseYValidation, DiseaseYPred_data);
       figure
       cm = confusionchart(Diseasem,Diseaseorder);
end

end
