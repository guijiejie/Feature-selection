function [out] = fsTtest(X,Y)
[m,n] = size(X);
W = zeros(1,n);
c = length(unique(Y));

for i=1:n
    temp = 0;
    for k=1:c
        for j = (k+1):c
            X1 = X(Y == k,i);
            X2 = X(Y == j,i);
            
            n1 = size(X1,1);
            n2 = size(X2,1);
            
            mean_X1 = sum(X1)/n1;
            mean_X2 = sum(X2)/n2 ;
            
            S_X1 = sum((X1 - mean_X1).^2);
            S_X2 = sum((X2 - mean_X2).^2);
            Sw = sqrt((S_X1+S_X2)/(n1+n2-2));
            if Sw ==0
                Sw = eps;
            end
            
            temp = temp+abs(mean_X1 - mean_X2)/( Sw*sqrt( (1/n1)+ (1/n2) ));
        end
    end
    W(1,i) = temp;
end

[foo out.fList] = sort(W, 'descend');
out.W = W;
out.prf = -1;
end
