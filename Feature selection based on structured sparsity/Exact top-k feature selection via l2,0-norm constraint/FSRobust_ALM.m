% min_{||W||_20=k}  ||X'*W+1*b'-Y||_21
function [feature_idx, W, b, obj] = FSRobust_ALM(X, Y, k, mu, rho, NITER)
% X: d*n data matrix, each column is a data point
% Y: n*c label matrix, Y(i,j)=1 if xi is labeled to j, and Y(i,j)=0 otherwise
% k: number of selected features
% mu, rho: parameters in the ALM optimization method
% NITER: iteration number
% feature_idx: indices of selected features
% W: d*c embedding matrix
% b: c*1 bias vector
% obj: objective values in the iterations

% Ref:
% Xiao Cai, Feiping Nie, Heng Huang. 
% Exact Top-k Feature Selection via l2,0-Norm Constraint. 
% The 23rd International Joint Conference on Artificial Intelligence (IJCAI), 2013.


obj = zeros(NITER,1);

[d, n] = size(X);
c = size(Y,2);
Xm = X-mean(X,2)*ones(1,n);

Lambda = zeros(d,c);
Sigma = zeros(n,c);
V = rand(d,c);
E = rand(n,c);
W = V;

inXX = Xm*inv(Xm'*Xm+eye(n));
for iter = 1:NITER
    inmu = 1/mu;
    tem = Y+E-inmu*Sigma;
    b = mean(tem)';
    V1 = (V-inmu*Lambda+Xm*tem); W = V1 - inXX*(Xm'*V1);
    %Wg = XX*W - (V-inmu*Lambda+Xm*tem); st = trace(Wg'*Wg)/trace(Wg'*(XX*Wg)); W = W - st*Wg;
    
    WL = W+inmu*Lambda;
    w = sum(WL.*WL,2);
    [~, idx] = sort(-w);
    V = zeros(d,c);
    V(idx(1:k),:) = WL(idx(1:k),:);
    
    XW = Xm'*W+ones(n,1)*b'-Y;
    XWY = XW+inmu*Sigma;
    for i = 1:n
        w = XWY(i,:);
        la = sqrt(w*w');
        lam = 0;
        if la > inmu
            lam = 1-inmu/la;
        elseif la < -inmu
            lam = 1+inmu/la;
        end;
        E(i,:) = lam*w;
    end;
    
    Lambda = Lambda + mu*(W-V);
    Sigma = Sigma + mu*(XW-E);   
    mu = min(10^10,rho*mu);

    err = Xm'*V+ones(n,1)*b'-Y;
    obj(iter) = sum(sqrt(sum(err.*err,2)));
end;

feature_idx = sort(idx(1:k));
