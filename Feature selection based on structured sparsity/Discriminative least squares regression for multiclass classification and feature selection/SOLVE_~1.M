% CORRESPONDENCE INFORMATION
%    This code is written by Shiming Xiang and Feiping Nie

%     Shiming Xiang :  National Laboratory of Pattern Recognition,  Institute of Automation, Academy of Sciences, Beijing 100190
%                                   Email:   smxiang@gmail.com
%     Feiping Nie:        Department of Computer    Science and Engineering,   University of Texas at Arlington, Arlington, TX 76019 USA 
%                                   Email:   feipingnie@gmail.com


%    Comments and bug reports are welcome.  Email to feipingnie@gmail.com  OR  smxiang@gmail.com

%   WORK SETTING:
%    This code has been compiled and tested by using matlab    7.0

%  For more detials, please see the manuscript:
%   Shiming Xiang, Feiping Nie, Gaofeng Meng, Chunhong Pan, and Changshui Zhang. 
%   Discriminative Least Squares Regression for Multiclass Classification and Feature Selection. 
%   IEEE Transactions on Neural Netwrok and Learning System (T-NNLS),  volumn 23, issue 11, pages 1738-1754, 2012.

%   Last Modified: Nov. 2, 2012, By Shiming Xiang


function [X, obj] = solve_iteratively_L21(A, Y)
%% minimize 21-norm with equality constraints

% Solve the following equivalent problem: 
%  min_X  ||X||_21
%  s.t.   A X = Y

n = size(A,  2);
m = size(A,  1);

ITER = 10;
obj = zeros(ITER,1);
d = ones(n, 1);                                              % initialization

epsilon = 10^-5;
obj1 = -1000;

for iter = 1 : ITER
    D = spdiags(d, 0, n, n);
    lambda = ((A * D) * A') \ Y;
    X = D *(A' * lambda);
    d = sqrt(sum(X .* X,2))  + 0.00000001;   
    % d = sqrt(sum(X .* X,2));  
     
    obj(iter) = sum(d);
    
    if abs( obj(iter) -  obj1) < epsilon
        break;
    end
    obj1 = obj(iter);
end


 return;