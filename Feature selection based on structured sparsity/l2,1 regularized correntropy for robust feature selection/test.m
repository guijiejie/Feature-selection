%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Description
%  A classification problem with three classes
%Code
%  1. Robust feature selection via compound norms minization (L21 and
%  frobenius norms)
%  2. Robust feature selection via L21 reguarlized correntropy 
%
%Reference  
%  Ran He, Tieniu Tan, Liang Wang and Wei-Shi Zheng. 
%  L21 Regularized Correntropy for Robust Feature Selection. In IEEE CVPR,2012.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function test

    %Classification problem with three classes
    A= rand(50,300);
    B= rand(50,300)+2;
    C= rand(50,300)+3;
    
    % label vector for the three classes
    label=[ones(300,1);2*ones(300,1);3*ones(300,1)];
    data =[A B C];
    
    % generate a sparse matrix whose dimension is 600
    sdata =zeros(600,900);
    sdata(1:12:600,:) = data;
    
    %Robust feature selection via compound norms minization (L21 and frobenius norms)
    [W,feaind,dd,T1] = RFS(sdata,label,0.01);
    
    figure; % show select feature    
    plot(1:600,dd);
    legend(['Sparsity:' num2str(length(find(dd>0)))])
    %show the sparse data in a low 2D space
    figure;
    ds=W'*sdata;
    plot(ds(2,:),ds(3,:),'o');
    
    %Robust feature selection via L21 reguarlized correntropy
    [W,feaind,dd,T] = CRFS(sdata,label,0.01);
    figure; % show select feature
    plot(1:600,dd);
    legend(['Sparsity:' num2str(length(find(dd>0)))])
    %show the sparse data in a low 2D space
    figure;
    ds=W'*sdata;
    plot(ds(2,:),ds(3,:),'o');