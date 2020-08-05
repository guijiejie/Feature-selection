% CORRESPONDENCE INFORMATION
%    This code is written by Shiming Xiang and Feiping Nie

%     Shiming Xiang :  National Laboratory of Pattern Recognition,  Institute of Automation, Academy of Sciences, Beijing 100190
%                                   Email:   smxiang@gmail.com
%     Feiping Nie:       Department of Computer Science and Engineering,  University of Texas at Arlington, Arlington, TX 76019 USA 
%                                   Email:   feipingnie@gmail.com


%    Comments and bug reports are welcome.  Email to feipingnie@gmail.com  OR  smxiang@gmail.com

%   WORK SETTING:
%    This code has been compiled and tested by using matlab    7.0

%  For more detials, please see the manuscript:
%   Shiming Xiang, Feiping Nie, Gaofeng Meng, Chunhong Pan, and Changshui Zhang. 
%   Discriminative Least Squares Regression for Multiclass Classification and Feature Selection. 
%   IEEE Transactions on Neural Netwrok and Learning System (T-NNLS),  volumn 23, issue 11, pages 1738-1754, 2012.

%   Last Modified: Nov. 2, 2012, By Shiming Xiang


function M = optimize_m_matrix(P, B)

% P:              the residual matrix, each row is a residual vector
% B:              construction matrix related to class label, each row is a constructtion vector

% return:        The optimized matrix

N = size(P, 1);
num_class = size(B, 2);

M1 = zeros(N, num_class);

M = max( B .* P,  M1); 

return;