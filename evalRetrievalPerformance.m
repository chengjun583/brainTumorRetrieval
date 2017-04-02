function [mAP, mPati, mRati] = evalRetrievalPerformance(db, dbLabel,...
    q, qLabel)
% compute mAP, precision@10, and precision@20

% -db: database with columns corresponding to observations.
% -dbLabel: a row vector representing the labels of records in databse.
% -q: queries with coluns corresponding to observations.
% -qLabel: a row vector representing the labels of queries.

% mAP: mean average precision.
% mPati: a column vector representing mean precision at position i.
% mRati: a column vector representing mean recall at position i.

[~, N] = size(q);
[~, M] = size(db);
AP = zeros(N, 1);
Pati = zeros(N, M);
Rati = zeros(N, M);

for i=1:N
    relevance = dbLabel==qLabel(i);        
    dist = sum( bsxfun(@minus, db, q(:, i)).^2, 1 );
    [AP(i), Pati(i,:), Rati(i,:)] = averagePrecision(dist, relevance);
end
mAP = mean(AP);
mPati = mean(Pati)';
mRati = mean(Rati)';

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