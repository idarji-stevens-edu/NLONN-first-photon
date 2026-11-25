function [A, n_pix, purity, order] = masked_camera_purity(Ztr, ytr, Zte, yte, targets)
% MASKED_CAMERA_PURITY  Camera behind a mask that keeps the purest pixels (Appendix A).
%   Purity of a pixel: share of the training photons on it that come from its most common class.
%   For each target mean acceptance, the smallest purity-ordered set reaching it is kept; the camera
%   decides from the first photon that passes, by the class its pixel names (rule from training).
%   Z: pixels x images photon distributions. A(t, i): accuracy of test image i at target t.
    n_tr = numel(ytr);
    Pj = zeros(size(Ztr, 1), 10);
    for c = 0:9, Pj(:, c+1) = double(sum(Ztr(:, ytr == c), 2)) / n_tr; end
    Px = sum(Pj, 2);
    purity = max(Pj, [], 2) ./ max(Px, realmin);
    [~, order] = sort(purity, 'descend');
    acc_cum = cumsum(Px(order));                     % mean training acceptance of the first k pixels
    A = zeros(numel(targets), numel(yte)); n_pix = zeros(numel(targets), 1);
    Y = (yte == (0:9)');
    for t = 1:numel(targets)
        k = find(acc_cum >= targets(t) - 1e-12, 1); if isempty(k), k = numel(order); end
        keep = order(1:k); n_pix(t) = k;
        Qtr = double(Ztr(keep, :)); p = sum(Qtr, 1); Qtr = Qtr ./ max(p, realmin); Qtr(:, p <= 0) = 0;
        Qte = Zte(keep, :);         p = sum(Qte, 1); Qte = Qte ./ max(p, realmin); Qte(:, p <= 0) = 0;
        P1 = zeros(k, 10);
        for c = 0:9, P1(:, c+1) = sum(Qtr(:, ytr == c), 2); end
        [~, d1] = max(P1, [], 2);
        A(t, :) = sum(Qte .* Y(d1, :), 1);
    end
end
