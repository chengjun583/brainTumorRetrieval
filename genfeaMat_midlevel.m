function feaMat = genfeaMat_midlevel(params, type)
% Generate mid-level features (i.e. BoW and fisher vector)

% -params.cvind: indices of cross validation.
% -params.testInd: test ind
% -params.radius: radius of disk-shaped structuring element used to dilate
%   the tumor region.
% -params.partitionType: can be 'distBased' or 'intensityBased'.
% -params.nRegion: number of regions.
% -params.dicSize: the number of centroids of kmeans or the number of
%   Gaussians of GMM.

% -params.patSize: patch size.
% -params.sorting: boolean value

% -type: a string denote feature type, which can be 'bow' or 'fv'.

% -feaMat: feature matrix with columns corresponding to observations.

rng('default');
type = lower(type);

cvind        = params.cvind;
testInd      = params.testInd;
radius       = params.radius;
partitionType= params.partitionType;
nRegion      = params.nRegion;
dicSize      = params.dicSize;

locFeaParams.patSize    = params.patSize;
locFeaParams.sorting    = params.sorting;


%% compute local features for clustering
% compute local features
fprintf('computing local features\n');
N = 30e4; 
trainInd = cvind~=testInd;
a=1:3064;
trainFiles=a(trainInd);
        
lenTrainFiles = length(trainFiles);
temp = cell(1, lenTrainFiles);
parfor i = 1 : lenTrainFiles
    % 提取当前图像中的所有local feature descriptors
    strc  = load(['imageData\', num2str(trainFiles(i)), '.mat']);
    norIm = minMaxNormalize(strc.cjdata.image);
    se    = strel('disk', radius, 0);
    mask  = imdilate(strc.cjdata.tumorMask, se); 
    feas  = f_extractLocFeasFromROI(norIm, mask, locFeaParams);

    % 从feas中选1/5放到cluFeas中
    len     = round(size(feas, 2)/5);
    ind     = randperm(size(feas, 2), len);
    temp{i} = feas(:, ind);
end

% 从cluFeas中选择N个
cluFeas = cell2mat(temp);
ind2    = randperm(size(cluFeas, 2), N);
cluFeas = cluFeas(:, ind2);

%% compute feature matrix
if strcmpi(type, 'bow')

	fprintf('performing kmeans clustering\n');
    [ctrs, assign]=vl_kmeans(cluFeas, dicSize,...
                    'MaxNumIterations', 500, 'algorithm', 'elkan');

	% generate bow
	feaMat = zeros(dicSize*nRegion, 3064);
	parfor iFile = 1 : 3064
%         fprintf('computing BoW representation for image %d\n', iFile);
		
		% 提取当前图像中的所有local feature descriptors
        strc  = load(['imageData\', num2str(iFile), '.mat']);
        norIm = minMaxNormalize(strc.cjdata.image);
        se    = strel('disk', radius, 0);
        mask  = imdilate(strc.cjdata.tumorMask, se); 
        feas  = f_extractLocFeasFromROI(norIm, mask, locFeaParams);
		
		% compute region index
        regionInd = genRegionInd(norIm, mask, nRegion, partitionType);
		
		% compute BoW histogram per region		
		temp = cell(nRegion, 1);
        for j = 1 : nRegion
            [ind, ~] = knnsearch(ctrs', feas(:, regionInd == j)');   
			h = hist(ind, 1 : dicSize);	% h is a row vector		
			temp{j} = (h/norm(h))';
        end
        feaMat(:, iFile) = cell2mat(temp);
    end
		
	
elseif strcmpi(type, 'fv')       
    % PCA reduction such that the dimenstion reduced data can better
    % fit the diagonal covariance restriction of GMM.
    fprintf('applying a PCA on local features\n');
    [coeff, score, latent] = princomp(cluFeas');
    miu = mean(cluFeas, 2);
    explained = latent./sum(latent).*100;
    cum       = cumsum(explained);
    a         = find(cum >= 99);
    cluFeas   = score(:, 1 : a(1))';

    % learn GMM
    fprintf('learning GMM\n');
    [means, covariances, priors] = vl_gmm(cluFeas, dicSize);

    % compute fisher vectors for each images.
    feaMat = zeros(a(1) * 2 * dicSize * nRegion, 3064, 'single');
    parfor i = 1 : 3064
%         fprintf('computing fisher vector for image %d\n', i);
        % 提取当前图像中的所有local feature descriptors
        strc  = load(['imageData\', num2str(i), '.mat']);
        norIm = minMaxNormalize(strc.cjdata.image);
        se    = strel('disk', radius, 0);
        mask  = imdilate(strc.cjdata.tumorMask, se); 
        feas  = f_extractLocFeasFromROI(norIm, mask, locFeaParams);
        feas = coeff(:, 1 : a(1))' * bsxfun(@minus, feas, miu);
        
        % compute region index
        regionInd = genRegionInd(norIm, mask, nRegion, partitionType);
        
        % compute FV per region
        temp = cell(nRegion, 1);
        for j = 1 : nRegion
            temp{j} = vl_fisher( feas(:, regionInd == j),...
                means, covariances, priors, 'improved' );            
        end

        feaMat(:, i) = cell2mat(temp);
    end
else 
    error('TYPE must be ''bow'' or ''fv''.\n')
end