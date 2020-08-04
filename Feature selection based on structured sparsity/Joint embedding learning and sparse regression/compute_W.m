function [W] = compute_W(W,data,D_mhalf)

[nSmp,nFea] = size(data);

%%%%%%%%%%%%%%%%%%%% Normalize W
if nSmp < 5000
    tmpD_mhalf = repmat(D_mhalf,1,nSmp);
    W = (tmpD_mhalf.*W).*tmpD_mhalf';
    clear tmpD_mhalf;
else
    [i_idx,j_idx,v_idx] = find(W);
    v1_idx = zeros(size(v_idx));
    for i=1:length(v_idx)
        v1_idx(i) = v_idx(i)*D_mhalf(i_idx(i))*D_mhalf(j_idx(i));
    end
    W = sparse(i_idx,j_idx,v1_idx);
    clear i_idx j_idx v_idx v1_idx
end
W = (W+W')/2;