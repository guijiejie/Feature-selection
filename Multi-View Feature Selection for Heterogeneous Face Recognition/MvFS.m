function [C,fList] = MvFS(X_multiview,Label_multiview)
num_class = length(unique(Label_multiview{1}));% For CUFSF, num_class = 700.
num_view = size(X_multiview,2);% For CUFSF, num_view = 2.
%% ************** Sjr ********************************************
A = cell(num_view,3);
for vi = 1:num_view
    [Xi ci li] = GetEachClass(X_multiview{vi},Label_multiview{vi},'x');
    [Mi ci li] = GetEachClass(X_multiview{vi},Label_multiview{vi},'m');
    [Numi ci li] = GetEachClass(X_multiview{vi},Label_multiview{vi},'num');
    A{vi,1} = Xi;
    A{vi,2} = Mi;
    A{vi,3} = Numi;
end
Ni = zeros(1,num_class);
dim = zeros(1,num_view);
for i=1:num_view
    Ni = Ni + A{i,3};
    dim(i) = size(X_multiview{i},1);
end
%% *********** LDA SW *******************************************
Sw = zeros(sum(dim),sum(dim));
for i=1:num_view
    Numi = A{i,3};
    Mi  = A{i,2};
    for j=i:num_view
        Numj = A{j,3};
        Mj = A{j,2};
        Xj = A{j,1};
        sij = zeros(dim(i),dim(j));
        vij = zeros(dim(j),dim(j));
        for ci = 1:num_class
            sij = sij - (Numi(ci)*Numj(ci)/Ni(ci))*(Mi(:,ci)*Mj(:,ci)');
            vij = vij + Xj{ci} * (Xj{ci}');
        end
        if j==i
            sij = vij + sij;
        end
        
        Sw(sum(dim(1:i-1))+1:sum(dim(1:i)), sum(dim(1:j-1))+1:sum(dim(1:j))) = sij;
        Sw(sum(dim(1:j-1))+1:sum(dim(1:j)), sum(dim(1:i-1))+1:sum(dim(1:i))) = sij';
    end
end

%% *********** LDA SB *******************************************
Sb = zeros(sum(dim),sum(dim));
n = sum(Ni);

for i=1:num_view
    mi = sum(X_multiview{i},2);
    Mi  = A{i,2};
    Numi = A{i,3};
    for j=i:num_view
        Numj = A{j,3};
        Mj = A{j,2};
        sij = zeros(dim(i),dim(j));
        
        mj = sum(X_multiview{j},2);
        for ci = 1:num_class
            sij = sij + (Numi(ci)*Numj(ci)/Ni(ci))*(Mi(:,ci)*Mj(:,ci)');
        end
        sij = sij - mi*mj'/n;
        Sb(sum(dim(1:i-1))+1:sum(dim(1:i)), sum(dim(1:j-1))+1:sum(dim(1:j))) = sij;
        Sb(sum(dim(1:j-1))+1:sum(dim(1:j)), sum(dim(1:i-1))+1:sum(dim(1:i))) = sij';
    end
end

%% LDA
Sb = Sb.*num_view;

dim = size(X_multiview{1},1);
FeatureScore = zeros(1,dim);
for i=1:dim
    index = [i (i+dim)];
    if sum(sum(Sw(index,index)))==0
        FeatureScore(1,i)= 100;% The same as Fisher score
    else
        FeatureScore(1,i)= sum(sum(Sb(index,index)))/sum(sum(Sw(index,index)));
    end
end
[C, fList] = sort(FeatureScore, 'descend');
% The follow code is for the case that different views have different
% dimensions.
% View1Index = zeros(1,dim(1));
% View2Index = zeros(1,dim(2));
% [C,View2Index(1,1)] = max(max(score));
% temp = find(score==C);
% if length(temp)==1
%     View1Index(1,1) = ceil(temp/dim(2));
% else
% end
% clear temp;
fprintf('MvFS finished\n');
