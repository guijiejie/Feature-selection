function [X, obj]=L21R21(A, Y, r)
%% 21-norm loss with 21-norm regularization

%% Problem
%
%  min_X  || A X - Y||_21 + r * ||X||_21       is equivalent to:
%
%  min_X  ||X||_21 + ||E||_21
%  s.t.   A X + r*E = Y

% Ref: Feiping Nie, Heng Huang, Xiao Cai, Chris Ding. 
% Efficient and Robust Feature Selection via Joint L21-Norms Minimization.  
% Advances in Neural Information Processing Systems 23 (NIPS), 2010.





[m n] = size(A);

[X, obj] = O21EC_inv([A, r*eye(m)], Y);
X = X(1:n,:);
obj = r*obj;



function [X, obj]=O21EC_inv(A, Y)
%% minimize 21-norm with equality constraints
% the row of A should be smaller than the column of A

%% Problem
%
%  min_X  ||X||_21
%  s.t.   A X = Y



[n] = size(A,2);

ITER = 50;
obj = zeros(ITER,1);
d = ones(n,1);   % initialization
for iter = 1:ITER
    D = spdiags(d,0,n,n);
    lambda = (A*D*A')\Y;
    X = D*(A'*lambda);
    d = sqrt(sum(X.*X,2));
    
    obj(iter) = sum(d);
end;
