clc;clear all;close all;
% load data
load('x_train.mat');
load('y_train.mat');
load('x_test.mat');
load('y_test.mat');

val_ratio = 0.2;
percent = round(size(x_train, 1) * (1 - val_ratio));

% split train data and validation data
train_data = x_train(1:percent, :);
label_data = categorical(y_train(1:percent));
val_data = x_train(percent + 1:end, :);
val_label = categorical(y_train(percent + 1:end));

class = categorical(y_train);
classnames = categories(class);

numFeatures = size(train_data,2);
numClasses = numel(classnames);

% define model
layers = [
    featureInputLayer(numFeatures)
    fullyConnectedLayer(4)
    reluLayer
    fullyConnectedLayer(8)
    reluLayer
    fullyConnectedLayer(4)
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

InitialLearnRate = 0.1;
MiniBatchSize = 10;
MaxEpochs = 30;
options = trainingOptions('adam', ...
    'MaxEpochs', MaxEpochs, ...
    'MiniBatchSize', MiniBatchSize, ...
    'Plots','training-progress', ...
    'InitialLearnRate', InitialLearnRate, ...
    'Verbose', true, ...
    'Shuffle', 'every-epoch', ...
    'ExecutionEnvironment', 'gpu',...
    'ValidationData',{val_data, val_label});

net = trainNetwork(train_data, label_data, layers, options);
predict_class = predict(net, x_test, 'MiniBatchSize', MiniBatchSize);

figure;
scatter(x_test(find(y_test==1),1),x_test(find(y_test==1),2),'filled');
hold on;
scatter(x_test(find(y_test==2),1),x_test(find(y_test==2),2),'filled');
title("Test data");

class1 = YPred(:,1);
class2 = YPred(:,2);
figure;
scatter(x_test(find(class1 > 0.5),1),x_test(find(class1 > 0.5),2),'filled');
hold on;
scatter(x_test(find(class2 > 0.5),1),x_test(find(class2 > 0.5),2),'filled');
title("Predict data");

