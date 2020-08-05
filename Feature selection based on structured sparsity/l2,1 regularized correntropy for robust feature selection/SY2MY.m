function Y = SY2MY(Y)
%%%%%%%%%%%%%%%%%%%%%%%%
%   Description:
%   Transform Y with single column to Y with the matrix representation.
%
%   Parameters:
%
%   Y - Single column, marked by 1 - k.
%
%   Output:
%
%   Y - with matrix representation, has k columns the column with +1 is
%   the affiliation of the the instances.
%

numI = size(Y,1);

if size(Y,2) < 2
    u = unique(Y);
    k = length(u);
    
    if k >= 2
        newY = -ones(numI,k);
        for i = 1:length(u);
            newY(Y==u(i),u(i)) = 1;
        end
    else
        
    end
    Y = newY;
end