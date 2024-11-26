clc;
clear all;
close all;
close all hidden;
warning off;
%% Selection Dialogue Box
message = "Choose your Selection of Work";
options = [
    "Diabetic Retinopathy diagnosis using Fundus Images", 
    "Age-Related Macular Degeneration Diagnosis Using OCT Images"
];
choice = menu(message, options);

if choice == 1 
    disp("Diabetic Retinopathy diagnosis using Fundus Images Classification has Started");
    Diabetic_Retinopathy; % Ensure this function is defined elsewhere in your code
elseif choice == 2
    disp("Age-Related Macular Degeneration Diagnosis Using OCT Images Classification has Started");
    Age_Related_Macular_Degeneration; % Ensure this function is defined elsewhere in your code
else
    disp("No selection made. Please choose an option.");
end
