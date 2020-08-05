% CORRESPONDENCE INFORMATION
%    This code is written by Shiming Xiang and Feiping Nie

%     Shiming Xiang :  National Laboratory of Pattern Recognition,  Institute of Automation, Academy of Sciences, Beijing 100190
%                                   Email:   smxiang@gmail.com
%     Feiping Nie:        Department of Computer  Science and Engineering,   University of Texas at Arlington, Arlington, TX 76019 USA 
%                                   Email:   feipingnie@gmail.com


%    Comments and bug reports are welcome.  Email to feipingnie@gmail.com  OR  smxiang@gmail.com

%   WORK SETTING:
%    This code has been compiled and tested by using matlab    7.0 and    R2010a

%  For more detials, please see the manuscript:
%   Shiming Xiang, Feiping Nie, Gaofeng Meng, Chunhong Pan, and Changshui Zhang. 
%   Discriminative Least Squares Regression for Multiclass Classification and Feature Selection. 
%   IEEE Transactions on Neural Netwrok and Learning System (T-NNLS),  volumn 23, issue 11, pages 1738-1754, 2012.

%   Last Modified: Nov. 2, 2012, By Shiming Xiang


% =========================================================================
% A Demo  Example to run the code for feature selection 
% =========================================================================


load data_XX;         % the training data,  each column is a data point
load data_Y_id;      %class IDs:     a column vector, a column vector,  such as  [1, 2, 3, 4, 1, 3, 2, ...]'

lambda_para  = 1.0;
u_para = 1000;
iters = 30;
epsilon = 0.0001;


% The first step: Train the model
tic
[W, b] = train_feature_selection(XX, Y_id, lambda_para, u_para, iters, epsilon);
toc

%   The second step:  Select the features and output the selected features

WW = W .^ 2;
W_weight = sum(WW, 2);                                                                               % sum the element row-by-row
[Weight, index_sorted_features] = sort(-W_weight);                                   %  sort them from the largest to the smallest

num_selected_features = 10;                                                                         % for example, we want to select ten features among all of the source features 

% output the features
index_features_finally_seelcted = index_sorted_features(1 : num_selected_features);

% perform other tasks, ...........


return;

             
