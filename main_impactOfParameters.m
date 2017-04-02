% You can change the main 5 parameters to see their impact on retrieval 
% performance. See our paper for details

% 5 parameters you can change in this file:
% --params.radius, the radius of disk-shaped structuring element used to 
%   dilate the tumor region
% --params.nRegion, the number of pooling regions created by the
%   intensity order-based division method
% --ps, the size of raw image patches that are used as local features 
% --ds, the number of vocabulary/dictionary size 
% --dim, the reduced dimensionality in new space induced by the
%   projection matrix learned in CFML 

% Also, we compared bag-of-words and Fisher vector. You scan switch these
% two representations by changes the parameter to either 'bow' or 'fv' (line 46).

%  Each element of cell array 'mapCell' is a vector of 5 elements, containing
%  the retrieval performance (mAP) of each round of 5-fold cross validation.

clear
clc
tic

load cvind5fold.mat
load label.mat

params.cvind   = cvind;
params.radius  = 0;
params.partitionType = 'intensityBased'; 
params.nRegion = 1;
params.sorting = 0;

nt = 1;
val = [1 2 3 4 5]; % ²âÊÔ¼¯Ë÷Òý
ps = 7; %[5, 7];
ds = 64;
mapCell = cell(length(ps), length(ds));
for ips = 1 : length(ps)
for ids = 1 : length(ds)
    tic
    params.patSize = ps(ips);
    params.dicSize = ds(ids);
    
    for i = 1 : length(val) % 5-fold cross validation
        fprintf('i=%d*************************\n', i);
        params.testInd = val(i);

        % generate feature representations
        X = genfeaMat_midlevel(params, 'fv');

        
        % generate training and test set
        fprintf('learning projection matrix\n');
        trBoolInd = cvind ~= params.testInd;
        train = X(:, trBoolInd);
        trLabel = label(trBoolInd)';
        
        teBoolInd = cvind == params.testInd;
        test = X(:, teBoolInd);
        teLabel = label(teBoolInd);
        
        % learn projection matrix
        dmlMethod = 'cfml';        
        T = learnProjMat(train, trLabel, dmlMethod);
        
        % evaluate performance
        fprintf('evaluating retrieval performance\n');
        dim = 2; %1:10;
        for j = 1 : length(dim)
            W = T(:, 1:dim(j));
            [mAP(i, j), mPati, mRati] = evalRetrievalPerformance(W'*train, trLabel,...
                W'*test, teLabel);
        end
    end
    mapCell{ips, ids} = mAP;
    t(nt) = toc;
    nt = nt + 1;
end
end
toc