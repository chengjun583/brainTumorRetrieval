function W = learnProjMat(X, XLabel, methodType)
% learn projection matrix

% -X: feature matrix with columns corresponding to observations
% -XLabel: a row vector of labels.
% -methodType: a string denoting the type of distance metric learning 
%   method.

% -W: the learned projection matrix with columns correspond to prejection
%   vectors.

methodType = lower(methodType);

switch methodType
    case 'cfml'
        lambda = 1.5e-4;
        W = DML_CFML(X, XLabel, lambda);
    case 'fda'
        [~, W] = DML_FDA(X, XLabel');    
    otherwise
        error('undefined methodType\n');
end