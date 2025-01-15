clc
clear
close all

tic
%% 读取数据
baseFolderPath = 'Your Folder'; % 输入文件夹
outputPath = 'Your Folder'; % 输出文件夹
numLevels = 5;
Data = cell(numLevels, 1);
imageNum = 100;

for i = 1:numLevels
    folderPath = fullfile(baseFolderPath, num2str(i));
    imageFiles = dir(fullfile(folderPath, '*.jpg'));
    numFiles = length(imageFiles);
    
    images = cell(numFiles, 1);
    for j = 1:numFiles
        filePath = fullfile(folderPath, imageFiles(j).name);
        images{j} = imread(filePath);
    end
    
    Data{i} = images;
end

%% 数据切割
NumofEachCata = numFiles;  % 假设每个类别的图像数目相同
for i = 1:numLevels
    switch i
        case 1
            Level1 = Data{i}(1:NumofEachCata);
        case 2
            Level2 = Data{i}(1:NumofEachCata);
        case 3
            Level3 = Data{i}(1:NumofEachCata);
        case 4
            Level4 = Data{i}(1:NumofEachCata);
        case 5
            Level5 = Data{i}(1:NumofEachCata);
    end
end

%% 数据预处理 - 灰度共生矩阵
% 灰度处理
Level1 = glcmImage(Level1);
Level2 = glcmImage(Level2);
Level3 = glcmImage(Level3);
Level4 = glcmImage(Level4);
Level5 = glcmImage(Level5);

%% 保存灰度共生矩阵图像和相关度
% saveGLCMImagesAndCorrelation(Level1, fullfile(outputPath, 'Level1'));
% saveGLCMImagesAndCorrelation(Level2, fullfile(outputPath, 'Level2'));
% saveGLCMImagesAndCorrelation(Level3, fullfile(outputPath, 'Level3'));
% saveGLCMImagesAndCorrelation(Level4, fullfile(outputPath, 'Level4'));
% saveGLCMImagesAndCorrelation(Level5, fullfile(outputPath, 'Level5'));

saveGLCMImagesAndFeatures(Level1, fullfile(outputPath, 'Level1'));
saveGLCMImagesAndFeatures(Level2, fullfile(outputPath, 'Level2'));
saveGLCMImagesAndFeatures(Level3, fullfile(outputPath, 'Level3'));
saveGLCMImagesAndFeatures(Level4, fullfile(outputPath, 'Level4'));
saveGLCMImagesAndFeatures(Level5, fullfile(outputPath, 'Level5'));
toc

function glcm = glcmImage(file)
    for i = 1:length(file)
        img = file{i};
        glcm = im2gray(img);
        glcm = graycomatrix(glcm, 'NumLevels', 64, 'Offset', [-1, -1], 'Symmetric', true);
        file{i} = glcm;
    end
    glcm = file;
end

function saveGLCMImagesAndFeatures(glcmData, folderPath)
    if ~exist(folderPath, 'dir')
        mkdir(folderPath);
    end
    numImages = length(glcmData);
    energy = zeros(numImages, 1);
    contrast = zeros(numImages, 1);
    correlation = zeros(numImages, 1);
    homogeneity = zeros(numImages, 1);
    entropy = zeros(numImages, 1);

    for i = 1:numImages
        img = mat2gray(glcmData{i});
%         将灰度图像转换为 RGB 图像
        img_rgb = cat(1, img, img, img);
%         调整图像大小为 224x224
        img_resized = imresize(img, [224 224]);
        imwrite(img_resized, fullfile(folderPath, ['image_' num2str(i) '.png']));
        
        % 计算纹理特征
        stats = graycoprops(glcmData{i}, {'Energy', 'Contrast', 'Correlation', 'Homogeneity'});
        energy(i) = stats.Energy;
        contrast(i) = stats.Contrast;
        correlation(i) = stats.Correlation;
        homogeneity(i) = stats.Homogeneity;
        
        % 计算熵
        P = glcmData{i} ./ sum(glcmData{i}(:));
        P(P == 0) = []; % 忽略零元素
        entropy(i) = -sum(P .* log2(P));
    end

    % 保存纹理特征
    save(fullfile(folderPath, 'features.mat'), 'energy', 'contrast', 'correlation', 'homogeneity', 'entropy');
end

% % function saveGLCMImagesAndCorrelation(glcmData, folderPath)
% %     if ~exist(folderPath, 'dir')
% %         mkdir(folderPath);
% %     end
% %     correlations = zeros(length(glcmData), 1);
% %     for i = 1:length(glcmData)
% %         img = mat2gray(glcmData{i});
% %         % 将灰度图像转换为 RGB 图像
% %         img_rgb = cat(3, img, img, img);
% %         % 调整图像大小为 224x224
% %         img_resized = imresize(img_rgb, [224 224]);
% %         imwrite(img_resized, fullfile(folderPath, ['image_' num2str(i) '.png']));
% %         
% %         % 计算相关度
% %         stats = graycoprops(glcmData{i}, 'Correlation');
% %         correlations(i) = stats.Correlation;
% %     end
% %     save(fullfile(folderPath, 'correlations.mat'), 'correlations');
% % end