% CORRESPONDENCE INFORMATION
%    This code is written by Shiming Xiang and Feiping Nie

%     Shiming Xiang :  National Laboratory of Pattern Recognition,  Institute of Automation, Academy of Sciences, Beijing 100190
%                                   Email:   smxiang@gmail.com
%     Feiping Nie:       Department of Computer  Science and Engineering,   University of Texas at Arlington, Arlington, TX 76019 USA 
%                                   Email:   feipingnie@gmail.com


%    Comments and bug reports are welcome.  Email to feipingnie@gmail.com  OR  smxiang@gmail.com

%   WORK SETTING:
%    This code has been compiled and tested by using matlab    7.0

%  For more detials, please see the manuscript:
%   Shiming Xiang, Feiping Nie, Gaofeng Meng, Chunhong Pan, and Changshui Zhang. 
%   Discriminative Least Squares Regression for Multiclass Classification and Feature Selection. 
%   IEEE Transactions on Neural Netwrok and Learning System (T-NNLS),  volumn 23, issue 11, pages 1738-1754, 2012.

%   Last Modified: Nov. 2, 2012, By Shiming Xiang

function [W, b] = train_feature_selection(X, class_id, lambda_para, u_para, iters, epsilon)

% X:                            each column is a data point
% class_id:                a column vector, a column vector,  such as  [1, 2, 3, 4, 1, 3, 2, ...]'
% lambda_para:      The lambda parameter in Equation (29), in the paper 
% u_para:                  the parameter c in the theorem, it is a very large positive number, generally, it is infinite. 
% iters:                       the largest number of iterations
% epsilon:                  For convergence  control

[dim, N] = size(X);
num_class = max(class_id);

Y = zeros(num_class, N);
B = -1 * ones(N, num_class);


for i = 1 : N
    Y( class_id(i),  i) = 1.0;  
    B(i, class_id(i) )  = 1.0;
end

[W0, b0] = least_squares_regression(X,  Y,  lambda_para);   % Here we use the soultion to the standard least squares regreesion as the initial solution  
W = W0;   
b = b0;

XX = [X; u_para * ones(1, N)];                                                       % construct the new training data by using homogeneous coordinates 

for i = 1: iters
    
    %first, optimize matrix M.
    P = X' * W0 + ones(N, 1) * b0' - Y';      % each row is a residual  vector
    
    M = optimize_m_matrix(P, B);
    
    % optimize W and b by using the theorem
    R  = Y' + (B .* M);
    [TT, obj] = optimize_L21(XX', R, lambda_para);
    W = TT(1:dim, :);
    b =    u_para * TT(dim + 1, :)';      % According to Lemma 1  
    
    if ( trace ( (W - W0)' * (W - W0) )  + ( b - b0)' *  (b - b0)   <  epsilon)     % check if it reaches the convergence point 
        break;
    end
    
    W0 = W;
    b0 = b;
      
end


return;