%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Description
%  Robust feature selection via compound norms minization (L21 and frobenius norms)
%    min_{U,E} ||U||_{21} + ||E||_F s.t. X^TU+\lambda E=Y
%  smooth relaxation
%    min_{U,E} \sum_i {\sqrt{\varepsilon+||u^i||_2^2}} + ||E||_F s.t.   X^T U + \lambda E =Y
%Input
%  Data   d*n data matrix
%  label  n*1 label vector
%  lambda regularization parameter
%Output
%  W      d*c projection matrix
%  feaind d*1 selected feature index vector
%  dd     d*1 weight vector
%  T1     CPU time
%Reference  
%  Ran He, Tieniu Tan, Liang Wang and Wei-Shi Zheng. 
%  L21 Regularized Correntropy for Robust Feature Selection. In IEEE CVPR,2012.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [W,feaind,dd,T1] = RFS(Data,label,lambda)
    T1 = cputime;
    Y = SY2MY(label);
    Y(find(Y==-1))=0;
    mn=size(Data,1);
    dd= ones(mn,1);
    iter =5;
    
    for i=1:iter
    
        % calculate (A'*D*A)
        d1 = dd(1:size(Data,1));
        T = Data'*(Data.*repmat(d1,1,size(Data,2)))+lambda*eye(size(Data,2));
        
        Z = T\Y;
        
        %calculate U
        U= (repmat(dd,1,size(Data,2)).*Data)*Z;

        %calculate the D
        dd = U.*U;
        dd = 2*sqrt(sum(dd'))';  
    end
    
    q1 = dd;
    W  = U;
    [v,feaind]=sort(q1,'descend');
    T1 = cputime -T1;

end