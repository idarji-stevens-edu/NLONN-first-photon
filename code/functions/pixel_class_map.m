function [d, S] = pixel_class_map(X, idx, label, mask, geo, kind, classes)
% PIXEL_CLASS_MAP  Equal-prior MAP class of each camera pixel, from the images X(:,:,idx).
%   label: labels of X; S: pixels x classes, summed photon probability per class
    S = zeros(geo.N^2, numel(classes));
    for s = 1:1000:numel(idx)
        i = idx(s:min(s + 999, end));
        P = camera_distributions(digit_fields(X(:, :, i), geo), mask, geo, kind);
        for q = 1:numel(classes), S(:, q) = S(:, q) + sum(P(:, label(i) == classes(q)), 2); end
    end
    n = arrayfun(@(c) nnz(label(idx) == c), classes);
    [~, q] = max(S ./ n, [], 2);
    d = classes(q);
end
