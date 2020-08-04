function Y = compute_Y(data, W, ReducedDim, D_mhalf)

[nSmp,nFea] = size(data); 

dimMatrix = size(W,2);
if (dimMatrix > 500 & ReducedDim < dimMatrix/10)
    option = struct('disp',0);
    [Y, eigvalue] = eigs(W,ReducedDim,'la',option);
    eigvalue = diag(eigvalue);
else
    W = full(W);
    [Y, eigvalue] = eig(W);
    eigvalue = diag(eigvalue); 

    [junk, index] = sort(-eigvalue);
    eigvalue = eigvalue(index);
    Y = Y(:,index);
    if ReducedDim < length(eigvalue)
        Y = Y(:, 1:ReducedDim);
        eigvalue = eigvalue(1:ReducedDim);
    end
end
    
eigIdx = find(abs(eigvalue) < 1e-6);
eigvalue (eigIdx) = [];
Y (:,eigIdx) = [];

nGotDim = length(eigvalue); 
    
idx = 1; 
while(abs(eigvalue(idx)-1) < 1e-12)
    idx = idx + 1;
    if idx > nGotDim
        break;
    end 
end
idx = idx - 1;

if(idx > 1)  
    % more than one eigenvector of 1 eigenvalue
    u = zeros(size(Y,1),idx);
    d_m = 1./D_mhalf;
    cc = 1/norm(d_m);
    u(:,1) = cc./D_mhalf;

    bDone = 0; 
    for i = 1:idx
        if abs(Y(:,i)' * u(:,1) - 1) < 1e-14
            Y(:,i) = Y(:,1);
            Y(:,1) = u(:,1);
            bDone = 1;
        end
    end

    if ~bDone
        for i = 2:idx
            u(:,i) = Y(:,i);
            for j= 1:i-1
                u(:,i) = u(:,i) - (u(:,j)' * Y(:,i))*u(:,j);
            end
            u(:,i) = u(:,i)/norm(u(:,i));
        end
        Y(:,1:idx) = u; 
    end
end
    
if nGotDim < 5000
    Y = repmat(D_mhalf,1,nGotDim).*Y;
else
    for k = 1:nGotDim
        Y(:,k) = Y(:,k).*D_mhalf;
    end
end
    
Y(:,1) = [];