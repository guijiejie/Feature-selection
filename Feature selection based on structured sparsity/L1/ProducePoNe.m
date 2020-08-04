function [y,A] = ProducePoNe(gnd,fea);
% Produce positive and negative examples
% This code is written by Jie Gui (guijie@ustc.edu).
%   V1 --July 25th, 2014
%   V2 --March 24th, 2015
[d,n] = size(fea);
nClass = max(gnd);%length(unique(gnd));
nEachClass = zeros(nClass,1);
nPos = 0; %the number of positive examples
nNeg = 0; %the number of negative examples

for i=1:nClass
    index = find(gnd == i);
    nEachClass(i,1) = length(index);
end

for i=1:nClass
    nPos = nPos+(nEachClass(i,1)*(nEachClass(i,1)-1))/2;
    nNeg = nNeg+nEachClass(i,1)*(n-nEachClass(i,1));
end

PosExm = zeros(d,nPos);%positive examples
NegExm = zeros(d,nNeg);%negative examples
nPosTemp = 0;
nNegTemp = 0;

for j = 1:n
    for k = (j+1):n
        temp_fea = fea(:,j) - fea(:,k);
        if gnd(j)==gnd(k)
            nPosTemp = nPosTemp+1;
            PosExm(:,nPosTemp) = temp_fea;
        else
            nNegTemp = nNegTemp+1;
            NegExm(:,nNegTemp) = temp_fea;
        end
    end
end

if nPos <= nNeg
    b = randperm(nNeg);
    b = b(1:nPos);
    NegExm = NegExm(:,b);
    A = [PosExm,NegExm];
    y = ones(nPos*2,1);
    y((nPos+1):(nPos*2),:) = -1;
else
    b = randperm(nPos);
    b = b(1:nNeg);
    PosExm = PosExm(:,b);
    A = [PosExm,NegExm];
    y = ones(nNeg*2,1);
    y((nNeg+1):(nNeg*2),:) = -1;
end
