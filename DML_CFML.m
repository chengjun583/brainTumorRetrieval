function [T,eigval,Ms,Md] = DML_CFML(X, Y, lambda)
% Regulized CFML

% -X: feature matrix with columns corresponding to observations.
% -Y: a row vector representing the label of X.
% -lambda: regularization parameter

% -T: prejection matrix with each column corresponding to projection
%   vector.

[d n]=size(X);
Ms = zeros(d,d);
Md = zeros(d,d);

nc = 0;
c=unique(Y);
for i=1:length(c)
    Xc = X(:,Y==c(i));
    
    nx = size(Xc,2);
    nc = nc + (nx - 1)*nx;
    
    Xc1 = sum(Xc,2);
    Xc1 = Xc1*Xc1';
    
    Xc2 = Xc*Xc';
    
    Ms = Ms + nx*Xc2 -Xc1;
end

Ms = Ms + Ms';

X1 = sum(X,2);
X1 = X1*X1';

X2 = X*X';
M = n*X2 - X1;
M = M + M';

Md = M - Ms;
nd = n*(n-1) - nc;

Ms = Ms./nc;
Md = Md./nd;


% [eigvec,eigval_matrix]=eig( (Ms + lambda*eye(d,d) )\Md);
[eigvec,eigval_matrix]=eig( Md, (Ms + lambda*eye(d,d)) );
eigval = real( diag(eigval_matrix) );
idx = find(eigval>0);
eigval = eigval(idx);
eigvec = eigvec(:,idx);
[eigval,sort_eigval_index]=sort(eigval,'descend');
T = eigvec(:,sort_eigval_index);
T = real(T);
T = T*diag( sqrt( eigval) );

