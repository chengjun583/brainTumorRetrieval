function [mAP, mPati, mRati] = evalRetrievalPerformance_types(db, dbLabel,...
    q, qLabel)
% compute mAP, precision@n for mixed types and specific type

% -db: database with columns corresponding to observations.
% -dbLabel: a row vector representing the labels of records in databse.
% -q: queries with coluns corresponding to observations.
% -qLabel: a row vector representing the labels of queries.

% mAP.all: a scalar representing mixed mAP
% mAP.specific: row vector, type-specific mAP
% mPati.all: a row vector representing mixed precision.
% mPati.specific: a c*n matrix, representing type-specific precision
%   c is the number of classes; n is the number of samples in db.
% mRati.all: a row vector representing mixed recall.
% mRati.specific: a c*n matrix, representing type-specific recall
%   c is the number of classes; n is the number of samples in db.

[~, N] = size(q);
[~, M] = size(db);
AP = zeros(N, 1);
Pati = zeros(N, M);
Rati = zeros(N, M);

uLabel = unique(qLabel);

for i=1:N
    relevance = dbLabel==qLabel(i);        
    dist = sum( bsxfun(@minus, db, q(:, i)).^2, 1 );
    [AP(i), Pati(i,:), Rati(i,:)] = averagePrecision(dist, relevance);
end

% for mixed class
mAP.all = mean(AP);
mPati.all = mean(Pati);
mRati.all = mean(Rati);

% for each specific class
for i = 1 : length(uLabel)
    mAP.specific(i) = mean( AP(qLabel == uLabel(i)) );    
    mPati.specific(i, :) = mean(Pati(qLabel == uLabel(i), :), 1);
    mRati.specific(i, :) = mean(Rati(qLabel == uLabel(i), :), 1);
end


function [AP, Pati, Rati] = averagePrecision(scores, rel)
% AveragePrecision 

[~, idx] = sort(scores);

rel = rel(idx);

% APatK = sum( rel(1:K) )./K;

Pati = cumsum(rel);
K=length(find(rel==1));
Rati=Pati/K;%查全率
Pati = Pati./[1:length(scores)];%返回1~length(scores)幅图像分别的查准率

AP = sum(rel.*Pati)./sum(rel);