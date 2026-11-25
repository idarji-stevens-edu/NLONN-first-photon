function [f, S] = mask_accuracy(w, Z, Y)
% MASK_ACCURACY  Accuracy of a camera behind transmission mask w, with its best per-pixel rule.
%   Z: pixels x images; Y: classes x images one-hot; S(k, c) = sum over class c of w_k Z(k, i) / p_i
    p = w' * Z; ip = zeros(size(p)); ip(p > 0) = 1 ./ p(p > 0);
    S = double((w .* Z) * (Y .* ip)');
    f = sum(max(S, [], 2)) / size(Z, 2);
end
