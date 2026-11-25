function [idx, acc_train, acc_cv, se_cv] = select_kernels(rate, label, classes)
% SELECT_KERNELS  Choose one kernel per class and estimate its accuracy by nested 5-fold CV.
%   rate(m, i): measured rate of library mask m on training digit i
%   The choice is repeated inside each training fold and scored on the held-out fold.
    n_folds = 5;
    fold = zeros(1, numel(label));
    for c = classes
        k = find(label == c); fold(k) = mod(0:numel(k) - 1, n_folds) + 1;   % class-balanced folds
    end
    p_cv = nan(1, numel(label));
    for f = 1:n_folds
        tr = fold ~= f; te = fold == f;
        s = choose_kernels(rate(:, tr), label(tr), classes);
        p_cv(te) = first_click_prob(rate(s, te), label(te), classes);
    end
    idx = choose_kernels(rate, label, classes);
    acc_train = mean(first_click_prob(rate(idx, :), label, classes));
    acc_cv = mean(p_cv);
    se_cv = std(p_cv) / sqrt(numel(p_cv));
end
