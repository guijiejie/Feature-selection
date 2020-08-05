% CORRESPONDENCE INFORMATION
%    This code is written by Shiming Xiang and Feiping Nie

%     Shiming Xiang :  National Laboratory of Pattern Recognition,  Institute of Automation, Academy of Sciences, Beijing 100190
%                                   Email:   smxiang@gmail.com
%     Feiping Nie:       Department of Computer Science and Engineering,   University of Texas at Arlington, Arlington, TX 76019 USA 
%                                   Email:   feipingnie@gmail.com


%    Comments and bug reports are welcome.  Email to feipingnie@gmail.com  OR  smxiang@gmail.com

%   WORK SETTING:
%    This code has been compiled and tested by using matlab    7.0

%  For more detials, please see the manuscript:
%   Shiming Xiang, Feiping Nie, Gaofeng Meng, Chunhong Pan, and Changshui Zhang. 
%   Discriminative Least Squares Regression for Multiclass Classification and Feature Selection. 
%   IEEE Transactions on Neural Netwrok and Learning System (T-NNLS),  volumn 23, issue 11, pages 1738-1754, 2012.

%   Last Modified: Nov. 2, 2012, By Shiming Xiang

function [X, obj] = optimize_L21(A, Y, lambda_para)
%% 21-norm loss with 21-norm regularization
%: each row is a data point

%  Note that: 
%  min_X  || A X - Y||_21 + lambda_para * ||X||_21       is equivalent to the following problem:

%  min_X  ||X||_21 + ||E||_21
%  s.t.   A X + lambda_para*E = Y





[m n] = size(A);
[X, obj] = solve_iteratively_L21([A, lambda_para * eye(m)], Y);

X = X(1:n, :);
obj = lambda_para * obj;
