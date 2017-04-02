function regionInd = genRegionInd(image, mask, nRegion, type)
% Generate region index of ROI defined by 'mask'.

% -image: image
% -mask: binary image denotes ROI
% -nRegion: partition ROI into nRegion regions
% -type: a string, which can be 'distBased' or 'intensityBased', denoting
%   the type of method used to partition ROI.

% -regionInd: a row vector of region index. the index can be the integer
%   from 1 to n if we partition the ROI into n regions.

type = lower(type);
switch type
    case 'distbased'
        e        = edge(mask); % 找到边界
        [er, ec] = find(e); 
        [r, c]   = find(mask);
        [idx, d] = knnsearch([er, ec], [r, c]);
        dist     = mapminmax(d', 0, 1); % 规范化到[0, 1]

        regionInd = zeros(size(dist));
        for i = 1 : nRegion
            ind = dist >= (i-1)/nRegion & dist <= i/nRegion;
            regionInd(ind) = i;
        end
        
    case 'intensitybased'
        % compute the intensity values of separatin points
        ints = double(image(mask));
        valSeparating = quantile(ints, linspace(0, 1, nRegion+1));

        % generate region index
        regionInd = zeros(size(ints));
        for i = 1 : nRegion
            ind = ints >=valSeparating(i) & ints<= valSeparating(i+1);
            regionInd(ind) = i;
        end        
        
    otherwise
        error('partitionType must be ''distBased'' or ''intensityBased''\n');
end 