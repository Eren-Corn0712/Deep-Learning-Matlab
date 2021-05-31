%% clear all data and close all figure
clc;close all;clear all;
%% load data
[XTrain, YTrain] = digitTrain4DArrayData;
[XTest, YTest] = digitTest4DArrayData;
idx = randperm(size(XTest, 4), 2000);
XValidation = XTest(:, :, :, idx);
XTest(:, :, :, idx) = [];
YValidation = YTest(idx);
YTest(idx) = [];
fprintf('X train size is  [%d x %d x %d x %d]\n',size(XTrain));
fprintf('X test size is  [%d x %d x %d x %d]\n',size(XTest));
fprintf('Y train size is  [%d x %d]\n',size(YTrain));
fprintf('Y test size is  [%d x %d]\n',size(YTest));
fprintf('X Validation size is  [%d x %d x %d x %d]\n',size(XValidation));
fprintf('Y Validation size is  [%d x %d]\n',size(YValidation));

%% show mnist
figure;
perm = randperm(size(XTrain, 4), 20);
for i = 1:20
    subplot(4,5,i);
    imshow(XTrain(:, :, :, perm(1, i)));
end
%% Layer define
layers = [
    imageInputLayer([28 28 1],"Name","imageinput")
    convolution2dLayer([7 7],48,"Name","conv_1")
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_1")
    convolution2dLayer([5 5],32,"Name","conv_2")
    batchNormalizationLayer("Name","batchnorm_2")
    reluLayer("Name","relu_2")
    maxPooling2dLayer([2 2],"Name","maxpool","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],32,"Name","conv_3")
    batchNormalizationLayer("Name","batchnorm_3")
    reluLayer("Name","relu_3")
    fullyConnectedLayer(10,"Name","fc_2")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
lgraph = layerGraph(layers);
analyzeNetwork(lgraph)
%% network train
InitialLearnRate = 0.001;
MiniBatchSize = 50;
MaxEpochs = 3;
options = trainingOptions('adam', ...
    'MaxEpochs', MaxEpochs, ...
    'MiniBatchSize', MiniBatchSize, ...
    'Plots','training-progress', ...
    'InitialLearnRate', InitialLearnRate, ...
    'Verbose', true, ...
    'Shuffle', 'every-epoch', ...
    'ExecutionEnvironment', 'gpu',...
    'ValidationData', {XValidation, YValidation},...
    'ValidationFrequency', 25);
net = trainNetwork(XTrain, YTrain, layers, options);
%% Predict
YPred = classify(net,XTest);
accuracy = sum(YPred == (YTest)) / numel(YTest);
fprintf('Accuracy = %2.4f %%\n', accuracy);
%% Confusion Matrix
figure;
confusionchart(YTest, YPred);
title('confusion Matrix');
