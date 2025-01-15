clc
clear
close all

a = logspace(-3, -5, 5);
b = linspace(1, 100, 5);
acc = [];
for i = 1:length(a)
    for j = 1:length(b)
        load resnet50_modi_image; %加载在ImageNet上预训练的网络模型
        imageInputSize = [224 224 3];%%数据维度
        %加载图像
        allImages = imageDatastore("Your Location", ...
            'IncludeSubfolders',true,...
            'LabelSource','foldernames');
        [training_set,validation_set] = splitEachLabel(allImages,0.3,'randomized');    %划分训练集30%和验证集70%

        augmented_training_set = augmentedImageSource(imageInputSize,training_set);
        augmented_validation_set = augmentedImageSource(imageInputSize,validation_set);

        opts = trainingOptions('adam', ...%%自适应动量的随机优化方法
            'MiniBatchSize', round(b),... % mini batch size, 批抽样个数
            'InitialLearnRate', a,... % fixed learning rate  学习率
            'LearnRateSchedule','piecewise',...%%采用学习率下降的方式
            'LearnRateDropFactor',0.25,...%%采用学习率下降的0.25
            'LearnRateDropPeriod',5,...%%学习率下降从第5次迭代下降
            'L2Regularization', 1e-4,... constraint%% 正则化参数
            'MaxEpochs',10,...%%%最大迭代次数
            'ExecutionEnvironment', 'gpu',...%%采用CPU训练
            'ValidationData', augmented_validation_set,...%%验证集（测试集）
            'ValidationFrequency',80,...%验证次数
            'ValidationPatience',8, ...
            'Plot','training-progress');%%显示训练过程

        net = trainNetwork(augmented_training_set, trainedNetwork_1.layerGraph, opts);%%训练网络

        [predLabels,predScores] = classify(net, augmented_validation_set);%%对测试数据进行分类
        plotconfusion(validation_set.Labels, predLabels)%%% 混淆矩阵画图
        PerItemAccuracy = mean(predLabels == validation_set.Labels);%计算正确率
        title(['overall per image accuracy ',num2str(round(100*PerItemAccuracy)),'%'])%显示正确率
        o = 1 - PerItemAccuracy;
        temp = 1 - o;
        acc = [temp, PerItemAccuracy];
    end
end
