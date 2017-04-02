function feas=f_extractLocFeasFromROI(im, mask, params)
% Extract square patches centered by each interest pixel of the mask

% -im: image.
% -mask: binary image with 1s representing the ROI.
% -params.sorting: boolean value
% -params.patSize: patch size.
% -feas: columns correspond to observations.

sorting = params.sorting;
patSize = params.patSize;

offset=(patSize-1)/2;
[r, c]=find(mask);

[x, y]=meshgrid(-offset:offset);
rDelta=y;
cDelta=x;        
feas=zeros(patSize^2, length(r), 'single');

n=0;
for i=1:patSize
    for j=1:patSize
        n=n+1;
        r1=r+rDelta(i, j);
        c1=c+cDelta(i, j);
        feas(n, :)=im(sub2ind(size(im), r1, c1));        
    end
end

% sorting
if sorting
    % sort patches
    temp = cell(offset+1, 1);

    % chessDistMat is like 
    % 2 2 2 2 2
    % 2 1 1 1 2
    % 2 1 0 1 2
    % 2 1 1 1 2
    % 2 2 2 2 2
    chessDistMat = 0;
    for i = 1 : offset
        chessDistMat = padarray(chessDistMat, [1, 1], i);
    end

    for i = 0 : offset
        ind = chessDistMat == i;
        temp{i+1}=sort(feas(ind, :), 1);
    end
    feas=cell2mat(temp);
end
    




