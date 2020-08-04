function [W_compute, Y, obj] = jelsr(data, W_ori, ReducedDim,alpha,beta)

%%%%%%%% Input: data: nSmp*nFea;
%%%             W_ori: The original local similarity matrix
%%%             ReducedDim: the dimensionality for low dimensionality
%%%                         embedding $Y$
%%%             alpha and beta ar two parameters

[nSmp,nFea] = size(data);

%%%%%%%%%%%%%%%%%%% Normalization of W_ori
D_mhalf = full(sum(W_ori,2).^-.5); 
W = compute_W(W_ori,data,D_mhalf); 

%%%%%%%%%%%%%%%%%% Eigen_decomposition
Y = compute_Y(data,W, ReducedDim, D_mhalf);       
if issparse(data)
    data = [data ones(size(data,1),1)];
    [nSmp,nFea] = size(data);
else
    sampleMean = mean(data);
    data = (data - repmat(sampleMean,nSmp,1));
end

%%% To minimize squared loss with L21 normalization
%%%%%%%%%%%% Initialization
AA = data'*data; 
Ay = data'*Y;
W_compute = (AA+alpha*eye(nFea))\Ay;
d = sqrt(sum(W_compute.*W_compute,2));

itermax = 20;
obj = zeros(itermax,1);

for iter = 1:itermax 
   %%%%%%%%%%%%%%%%%%% Fix D to updata W_compute, Y
   D = 2*spdiags(d,0,nFea,nFea);
   %%%%%%%%%%%%%%%% To updata Y
   A = (D*data'*data+alpha*eye(nFea));   
   Temp  = A\(D*data'); 
   Temp =  data*Temp;
   Temp = W_ori-beta*eye(nSmp)+beta*Temp; 
   
   %%%%% Normalization
   Temp = compute_W(Temp,data,D_mhalf); 
   %%%%% Eigen_decomposition   
   Y = compute_Y(data,Temp, ReducedDim, D_mhalf);
   
   %%%%%%%%%%%%%%%%% To updata W
   B = D*data'*Y; 
   W_compute = A\B;
   
   %%%%%%%%%%%%%%%%%% Fix W and update D
   d = sqrt(sum(W_compute.*W_compute,2));
   
end 
end 
 
