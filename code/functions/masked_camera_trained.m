function [w, d] = masked_camera_trained(Z, y)
% MASKED_CAMERA_TRAINED  Camera behind a trained per-pixel transmission mask (Appendix A).
%   The training images are split into a design and a validation half (alternating within each class).
%   1. Binary pixel sets: from the purity-ordered set at several mean acceptances, add or remove the
%      single pixel that most raises the design accuracy, keeping the set best on validation.
%   2. Transmissions: projected Adam on the design accuracy from the best set, w in [0, 1] with
%      max w = 1, keeping the step best on validation.
%   w: transmission of each pixel; d: class each pixel names (NaN where w = 0), from all training images.
    starts = [0.002 0.005 0.01 0.02 0.05 0.1];
    Z = single(Z); y = y(:)'; classes = unique(y); n_p = size(Z, 1);
    Y = (y == classes(:));
    h = false(size(y));
    for c = classes, ii = find(y == c); h(ii(1:2:end)) = true; end
    Zd = Z(:, h); Yd = Y(:, h); Zv = Z(:, ~h); Yv = Y(:, ~h);

    % 1. binary sets
    Pj = Zd * Yd' / size(Zd, 2); Px = sum(Pj, 2);
    [~, order] = sort(max(Pj, [], 2) ./ max(Px, realmin), 'descend'); acc_cum = cumsum(Px(order));
    best_val = -Inf;
    for s = 1:numel(starts)
        k = find(acc_cum >= starts(s) - 1e-12, 1); if isempty(k), k = n_p; end
        A = false(n_p, 1); A(order(1:k)) = true;
        fd = mask_accuracy(double(A), Zd, Yd); fv = mask_validation(double(A), Zd, Yd, Zv, Yv);
        A_best = A; v_best = fv;
        for it = 1:300
            [fa, ua, fr, ur] = mask_best_move(A, Zd, Yd);
            if max(fa, fr) <= fd + 1e-12, break, end
            if fa >= fr, A(ua) = true; fd = fa; else, A(ur) = false; fd = fr; end
            fv = mask_validation(double(A), Zd, Yd, Zv, Yv);
            if fv > v_best, v_best = fv; A_best = A; end
        end
        if v_best > best_val, best_val = v_best; best_A = A_best; end
    end
    w = double(best_A);

    % 2. continuous transmissions
    m1 = zeros(n_p, 1); m2 = zeros(n_p, 1); w_best = w; v_best = best_val; since = 0;
    for t = 1:1500
        [~, gr] = mask_gradient(w, Zd, Yd);
        m1 = 0.9 * m1 + (1 - 0.9) * gr; m2 = 0.999 * m2 + (1 - 0.999) * gr.^2;
        w = w + 0.02 * (m1 / (1 - 0.9^t)) ./ (sqrt(m2 / (1 - 0.999^t)) + 1e-12);
        w = max(w, 0); w = w / max(w);
        fv = mask_validation(w, Zd, Yd, Zv, Yv);
        if fv > v_best + 1e-12, v_best = fv; w_best = w; since = 0; else, since = since + 1; end
        if since >= 300, break, end
    end
    w = w_best; w(w < 1e-6) = 0;

    % rule from all training images
    [~, S] = mask_accuracy(w, Z, Y); [~, dk] = max(S, [], 2);
    d = classes(dk)'; d(w == 0) = NaN;
end
