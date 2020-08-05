function [X, obj]=L21R21_inv(A, Y, r, X0)
%% 21-norm loss with 21-norm regularization

%% Problem
%
%  min_X  || A X - Y||_21 + r * ||X||_21

% Ref: Feiping Nie, Heng Huang, Xiao Cai, Chris Ding. 
% Efficient and Robust Feature Selection via Joint L21-Norms Minimization.  
% Advances in Neural Information Processing Systems 23 (NIPS), 2010.



NIter = 50;
[m n] = size(A);
if nargin < 4
    d = ones(n,1);
    d1 = ones(m,1);
else
    Xi = sqrt(sum(X0.*X0,2));
    d = 2*Xi;
    AX = A*X0-Y;
    Xi1 = sqrt(sum(AX.*AX,2)+eps);
    d1 = 0.5./Xi1;
end;

if m>n
    for iter = 1:NIter
        D = spdiags(d,0,n,n);
        D1 = spdiags(d1,0,m,m);
        DAD = D*A'*D1;
        X = (DAD*A+r*eye(n))\(DAD*Y);

        Xi = sqrt(sum(X.*X,2));
        d = 2*Xi;

        AX = A*X-Y;
        Xi1 = sqrt(sum(AX.*AX,2)+eps);
        d1 = 0.5./Xi1;

        obj(iter) = sum(Xi1) + r*sum(Xi);
    end;
else
    for iter = 1:NIter
        D = spdiags(d,0,n,n);
        D1 = spdiags(d1,0,m,m);
        DAD = D*A'*D1;
        X = DAD*((A*DAD+r*eye(m))\Y);

        Xi = sqrt(sum(X.*X,2));
        d = 2*Xi;

        AX = A*X-Y;
        Xi1 = sqrt(sum(AX.*AX,2)+eps);
        d1 = 0.5./Xi1;

        obj(iter) = sum(Xi1) + r*sum(Xi);
    end;
end;
1;