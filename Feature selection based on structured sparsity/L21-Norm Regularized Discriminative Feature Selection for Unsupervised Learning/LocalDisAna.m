function L = LocalDisAna(X, para)
% unsupervised local discriminative analysis 
% each column is a data



[D, n] = size(X);

if isfield(para, 'k')
    k = para.k+1;
else
    k = 16;
end;
if isfield(para, 'lamda')
    lamda = para.lamda;
else
    lamda = 1000;
end;

Lc = eye(k) - 1/k*ones(k);
A = spalloc(n*k,n*k,5*n*k);
S = spalloc(n,n*k,5*n*k);
for i = 1:n
    dis = repmat(X(:,i),1,n) - X;
    dis = sum(dis.*dis);
    [dumb, nnidx] = sort(dis);
    Xi = X(:,nnidx(1:k));
    Xi = Xi*Lc;
    if D > k
        Ai = inv(lamda*eye(k) + Xi'*Xi);
        Ai = Lc*Ai*Lc;
    else
         Ai = Lc - lamda*Xi'*inv(eye(D) + lamda*Xi*Xi')*Xi;
    end;
    lidx = (i-1)*k+1:(i-1)*k+k;
    A(lidx, lidx) = Ai;
    S(nnidx(1:k),lidx) = eye(k);
end;
    
L = S*A*S';




