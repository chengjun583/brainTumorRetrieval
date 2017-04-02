function [Z,W] = DML_FDA(X,Y,r)
    
% Fisher Discriminant Analysis for Supervised Dimensionality Reduction
% It can work binary and multiclass. It also detects class label
% automatically and it returns W that can be used to multiply any new data
% sample for reduction e.g. X for testing. If you have any problem or 
% or difficulty, do not hesitate to send me an email.
% Usage:
%       
%       [Z,W]=FDA(X,Y,r)
%       or
%       [Z,W]=FDA(X,Y) , where r values by default is Number of classes
%       minus 1: C-1
% Input:
%    X:      d x n matrix of original samples
%            d --- dimensionality of original samples
%            n --- the number of samples 
%    Y:      n --- dimensional vector of class labels
%    r:      ----- dimensionality of reduced space (default:C-1)
%            r has to be from 1<=r<=C-1, where C is # of lables "classes"
% Output:
%    W:      d x r transformation matrix (Z=T'*X)
%    Z:      r x n matrix of dimensionality reduced samples 
%
%    (c) Sultan Alzahrani, PhD Student, Arizona State University.
%    ssalzahr@asu.edu,  http://www.public.asu.edu/~ssalzahr/
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
 
Z=0;
    W=0;
    [d,n]=size(X);
    [nY,dum]=size(Y);
  
    fprintf('\nNumber of instances: n = %d\n',n);
    fprintf('Dimensionality of original samples: d = %d\n',d);
    if(nY ~= n)
        fprintf('Error: Check Y: Y has to be a vector, nX1 = %d X 1\n\n',n);
        return
    end 
    i=1;
 
    ClsLbls=unique(Y');
    NumberOfClasses = size(ClsLbls,2);
    if(nargin==2)
        r=NumberOfClasses-1;
    end
    fprintf('Number of classes = d%\n',NumberOfClasses);
    fprintf('Class label list: %d\n',char(ClsLbls));
    fprintf('Dimensionality of reduced space: r = %d\n',r)
    disp('Please wait!');
   
    LocalMu = cell(1,NumberOfClasses);
    CovVal = cell(1,NumberOfClasses);
    sizeC=zeros(1,NumberOfClasses);
    % Compute local Mu, cov matrix, and number of observation of
    % for each class class
    for clsLbl=unique(Y')
        Xc=X(:,Y==clsLbl);
        LocalMu(i) = {mean(Xc,2)};
        CovVal(i) = {cov(Xc')};
        sizeC(i)=size(Xc,2);
        i=i+1;
    end
    
    
    
    %Computing  the Global Mu
    Global_Mu = zeros(d,1);
    
    for i = 1:NumberOfClasses
        Global_Mu = Global_Mu+LocalMu{i};
    end
    Global_Mu=Global_Mu./NumberOfClasses;
    %SB: Betweeness class scatter matrix
    %SW: Scatter Class Matrix
    
    SB = zeros(d,d);
    SW = zeros(d,d);
    
    for i = 1:NumberOfClasses
        SB = SB + sizeC(i).*(LocalMu{i}-Global_Mu)*(LocalMu{i}-Global_Mu)';
        SW = SW+CovVal{i};
    end
    
    % To reduce dimentionality, we need to find W that maximize the ratio
    % of Betweeness class Scatter to Within class Scatter.
    % W has to satisfy:
    % (1) Distance between the class means: the larger the better: SB
    % (2) The variance of each class: the smaller the better: SW
    % Thus J(W), the objective function is proportional to SB and
    % inversely proportional to Sw.
    % Thus invSw_by_SB is computed  and the projection w that maximizes 
    % this ratio. This problem is converted to an Eigen
    % vector problem for 1<=r<=C-1 
    % where W= [W1|W2|...|W_c-1] = argmax|W' SB W|/|W' SW W| => 
    %(SB-?_iSW)W_i=0
    
    invSw = inv(SW + 0.01*eye(size(SW)));
    invSw_by_SB = invSw * SB;

    [V,D] = eig(invSw_by_SB);
    eigval=diag(D);
    % Sort invSw_by_SB (which is a matrix of eigen vectors) and then select 
    % the top vectors assocaited with the top eigen values as follows
    
    % Sorting
    [sort_eigval,sort_eigval_index]=sort(eigval,'descend');
    % Selecting the top vectors
    W=V(:,sort_eigval_index(1:r));
   
    % Now Z has the dimentional reduced data sample X.
    Z = W'*X;
    [dZ,nZ]=size(Z);
    fprintf('Dimentionality reduction is done! and the new size data is %d X %d\n',dZ,nZ);
    
end