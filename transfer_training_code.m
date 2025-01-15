clc
clear
close all

a = logspace(-3, -5, 5);
b = linspace(1, 100, 5);
acc = [];
for i = 1:length(a)
    for j = 1:length(b)
        load resnet50_modi_image; %������ImageNet��Ԥѵ��������ģ��
        imageInputSize = [224 224 3];%%����ά��
        %����ͼ��
        allImages = imageDatastore("Your Location", ...
            'IncludeSubfolders',true,...
            'LabelSource','foldernames');
        [training_set,validation_set] = splitEachLabel(allImages,0.3,'randomized');    %����ѵ����30%����֤��70%

        augmented_training_set = augmentedImageSource(imageInputSize,training_set);
        augmented_validation_set = augmentedImageSource(imageInputSize,validation_set);

        opts = trainingOptions('adam', ...%%����Ӧ����������Ż�����
            'MiniBatchSize', 32,... % mini batch size, ����������
            'InitialLearnRate', 0.005,... % fixed learning rate  ѧϰ��
            'LearnRateSchedule','piecewise',...%%����ѧϰ���½��ķ�ʽ
            'LearnRateDropFactor',0.25,...%%����ѧϰ���½���0.25
            'LearnRateDropPeriod',5,...%%ѧϰ���½��ӵ�5�ε����½�
            'L2Regularization', 1e-4,... constraint%% ���򻯲���
            'MaxEpochs',10,...%%%����������
            'ExecutionEnvironment', 'gpu',...%%����CPUѵ��
            'ValidationData', augmented_validation_set,...%%��֤�������Լ���
            'ValidationFrequency',80,...%��֤����
            'ValidationPatience',8, ...
            'Plot','training-progress');%%��ʾѵ������

        net = trainNetwork(augmented_training_set, trainedNetwork_1.layerGraph, opts);%%ѵ������

        [predLabels,predScores] = classify(net, augmented_validation_set);%%�Բ������ݽ��з���
        plotconfusion(validation_set.Labels, predLabels)%%% ��������ͼ
        PerItemAccuracy = mean(predLabels == validation_set.Labels);%������ȷ��
        title(['overall per image accuracy ',num2str(round(100*PerItemAccuracy)),'%'])%��ʾ��ȷ��
        o = 1 - PerItemAccuracy;
        temp = 1 - o;
        acc = [temp, PerItemAccuracy];
    end
end