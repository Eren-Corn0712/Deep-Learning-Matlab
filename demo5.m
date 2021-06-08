%% clear all data and close all figure
clc;close all;clear;
%% load data
path = 'D:\大三\深度學習\demo\demo5\data\';
load([path, 'x_test.mat']);load([path, 'x_train.mat']);
load([path, 'y_test.mat']);load([path, 'y_train.mat']);

x_train = permute(x_train, [2, 3, 4, 1]);
x_test = permute(x_test, [2, 3, 4, 1]);
y_train = categorical(y_train); y_test = categorical(y_test);

classnames = categories(y_train);
numClasses = numel(classnames);

idx = randperm(size(x_test, 4), 20);
x_validation = x_test(:, :, :, idx);
x_test(:, :, :, idx) = [];
y_validation = y_test(idx);
y_test(idx) = [];

fprintf('x train size is  [%d x %d x %d x %d]\n',size(x_train));
fprintf('y train size is  [%d x %d]\n',size(y_train));
fprintf('x test size is  [%d x %d x %d x %d]\n',size(x_test));
fprintf('y test size is  [%d x %d]\n',size(y_test));
fprintf('x Validation size is  [%d x %d x %d x %d]\n',size(x_validation));
fprintf('y Validation size is  [%d x %d]\n',size(y_validation));
%% show data
figure;
perm = randperm(size(x_train, 4), 20);
for i = 1:20
    subplot(4,5,i);
    imshow(x_train(:, :, :, perm(1, i)));
end
%% Layer define
layers = [
 imageInputLayer([32 32 3],"Name","imageinput")
    convolution2dLayer([3 3],16,"Name","conv_1")
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_1")
    convolution2dLayer([3 3],16,"Name","conv_2")
    batchNormalizationLayer("Name","batchnorm_2")
    reluLayer("Name","relu_2")
    maxPooling2dLayer([2 2],"Name","maxpool","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],16,"Name","conv_3")
    batchNormalizationLayer("Name","batchnorm_3")
    reluLayer("Name","relu_3")
    convolution2dLayer([3 3],16,"Name","conv_4")
    batchNormalizationLayer("Name","batchnorm_4")
    reluLayer("Name","relu_4")
    maxPooling2dLayer([2 2],"Name","maxpoo2","Padding","same","Stride",[2 2])
    convolution2dLayer([2 2],16,"Name","conv_5")
    batchNormalizationLayer("Name","batchnorm_5")
    reluLayer("Name","relu_5")
    convolution2dLayer([2 2],16,"Name","conv_6")
    batchNormalizationLayer("Name","batchnorm_6")
    reluLayer("Name","relu_6")
    fullyConnectedLayer(128,"Name","fc_1")
    reluLayer("Name","relu_7")
    fullyConnectedLayer(numClasses,"Name","fc_2")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
lgraph = layerGraph(layers);
analyzeNetwork(lgraph);
%% Network training
MiniBatchSize = 128;
InitialLearnRate = 0.001;
valFrequency = floor(size(x_train,4)/MiniBatchSize);
MaxEpochs = 30;
options = trainingOptions('adam', ...
    'MaxEpochs', MaxEpochs, ...
    'MiniBatchSize', MiniBatchSize, ...
    'Plots','training-progress', ...
    'InitialLearnRate', InitialLearnRate, ...
    'Verbose', true, ...
    'Shuffle', 'every-epoch', ...
    'ExecutionEnvironment', 'gpu',...
    'ValidationData', {x_validation, y_validation},...
    'ValidationFrequency', valFrequency);
net = trainNetwork(x_train, y_train, layers, options);
%% Evaluate Trained Network
YPred = classify(net,x_test);
accuracy = sum(YPred == (y_test)) / numel(y_test);
fprintf('Accuracy = %2.4f %%\n', accuracy);

[YValPred,probs] = classify(net, x_validation);
validationError = mean(YValPred ~= y_validation);

YTrainPred = classify(net,x_train);
trainError = mean(YTrainPred ~= y_train);

disp("Training error: " + trainError*100 + "%");
disp("Validation error: " + validationError*100 + "%");

%% Confusion Matrix
figure('Units','normalized','Position',[0.2 0.2 0.4 0.4]);
cm = confusionchart(y_validation,YValPred);
cm.Title = 'Confusion Matrix for Validation Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

figure('Units','normalized','Position',[0.2 0.2 0.4 0.4]);
cm = confusionchart(y_test,YPred);
cm.Title = 'Confusion Matrix for Test Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
save('demo5');