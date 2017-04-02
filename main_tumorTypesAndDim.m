% retrieval performance for different tumor types and reduced dimensions

% You can change the main 5 parameters to see their impact on retrieval 
% performance. See our paper for details

% 5 parameters you may change in this file:
% --params.radius, the radius of disk-shaped structuring element used to 
%   dilate the tumor region
% --params.nRegion, the number of pooling regions created by the
%   intensity order-based division method
% --params.patSize, the size of raw image patches that are used as local features 
% --params.dicSize, the number of vocabulary/dictionary size 
% --dim, the reduced dimensionality in new space induced by the
%   projection matrix learned in CFML 

% Also, we compared bag-of-words and Fisher vector. You scan switch these
% two representations by changes the parameter to either 'bow' or 'fv' (line 46).


clear
clc
startVal = tic;

load cvind5fold.mat
load label.mat

params.cvind   = cvind;
params.radius  = 24;
params.partitionType = 'intensityBased'; 
params.nRegion = 8;
params.dicSize = 128;
params.patSize = 9;
params.sorting = 0;

val = [1 2 3 4 5]; % ²âÊÔ¼¯Ë÷Òý
dim = 1:10; 
lenDim = length(dim);
for i = 1 : length(val) % 5-fold cross validation
    tic
    
    fprintf('i=%d*************************\n', i);
    params.testInd = val(i);

    % generate feature representations
    X = genfeaMat_midlevel(params, 'fv');
    
    % generate training and test set 
    trBoolInd = cvind ~= params.testInd;
    train = X(:, trBoolInd);
    trLabel = label(trBoolInd)';
    
    teBoolInd = cvind == params.testInd;
    test = X(:, teBoolInd);
    teLabel = label(teBoolInd)';
    
    % learn projection matrix
    fprintf('learning projection matrix\n');
    dmlMethod = 'cfml';    
    T = learnProjMat(train, trLabel, dmlMethod);
    
    % evaluate performance
    fprintf('evaluating retrieval performance\n');
    for id = 1 : lenDim
        W = T(:, 1:dim(id));
        [mAP(i, id), mPati(i, id), mRati(i, id)] = evalRetrievalPerformance_types(W'*train, trLabel,...
            W'*test, teLabel);
    end
    
    toc
end

% compute mixed mAP, prec@10
mixedMAP = zeros(lenDim, 5);
mixedPrec10 = zeros(lenDim, 5);
for id = 1 : lenDim
for i = 1 : 5
    mixedMAP(id, i) = mAP(i, id).all;
    mixedPrec10(id, i) = mPati(i, id).all(10) ;
end
end
mixedMAP_mean = mean(mixedMAP, 2);
mixedMAP_std = std(mixedMAP, 1, 2);

mixedPrec10_mean = mean(mixedPrec10, 2); 
mixedPrec10_std = std(mixedPrec10, 1, 2);

% compute type-specific mAP, prec@10, prec@20
type1MAP = zeros(lenDim, 5);
type2MAP = zeros(lenDim, 5);
type3MAP = zeros(lenDim, 5);

type1Prec10 = zeros(lenDim, 5);
type2Prec10 = zeros(lenDim, 5);
type3Prec10 = zeros(lenDim, 5);

type1Prec20 = zeros(lenDim, 5);
type2Prec20 = zeros(lenDim, 5);
type3Prec20 = zeros(lenDim, 5);

for id = 1 : lenDim
    for i = 1 : 5
        type1MAP(id, i) = mAP(i, id).specific(1);
        type2MAP(id, i) = mAP(i, id).specific(2);
        type3MAP(id, i) = mAP(i, id).specific(3);
        
        type1Prec10(id, i) = mPati(i, id).specific(1, 10);
        type2Prec10(id, i) = mPati(i, id).specific(2, 10);
        type3Prec10(id, i) = mPati(i, id).specific(3, 10);
        
        type1Prec20(id, i) = mPati(i, id).specific(1, 20);
        type2Prec20(id, i) = mPati(i, id).specific(2, 20);
        type3Prec20(id, i) = mPati(i, id).specific(3, 20);
    end
end

type1MAP_mean = mean(type1MAP, 2);
type2MAP_mean = mean(type2MAP, 2);
type3MAP_mean = mean(type3MAP, 2);
type1MAP_std = std(type1MAP, 1, 2);
type2MAP_std = std(type2MAP, 1, 2);
type3MAP_std = std(type3MAP, 1, 2);

type1Prec10_mean = mean(type1Prec10, 2);
type2Prec10_mean = mean(type2Prec10, 2);
type3Prec10_mean = mean(type3Prec10, 2);
type1Prec10_std = std(type1Prec10, 1, 2);
type2Prec10_std = std(type2Prec10, 1, 2);
type3Prec10_std = std(type3Prec10, 1, 2);

type1Prec20_mean = mean(type1Prec20, 2);
type2Prec20_mean = mean(type2Prec20, 2);
type3Prec20_mean = mean(type3Prec20, 2);
type1Prec20_std = std(type1Prec20, 1, 2);
type2Prec20_std = std(type2Prec20, 1, 2);
type3Prec20_std = std(type3Prec20, 1, 2);


toc(startVal)