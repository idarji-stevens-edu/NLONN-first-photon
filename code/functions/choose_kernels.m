function idx = choose_kernels(rate, label, classes)
% CHOOSE_KERNELS  One mask per class from measured rates (Appendix B, Selection).
%   Each class slot starts from the mask with the largest lead for that class; then each slot in
%   turn takes the mask that most raises the training accuracy (up to 6 sweeps).
    rate = max(rate, 0);
    n_m = size(rate, 1); n_c = numel(classes);
    class_mean = zeros(n_m, n_c);
    for c = 1:n_c
        class_mean(:, c) = mean(rate(:, label == classes(c)), 2);
    end
    share = class_mean ./ max(sum(class_mean, 1), realmin);   % each class's profile over masks
    lead = share ./ max(sum(share, 2), realmin);               % a mask's lead for each class
    idx = zeros(1, n_c);
    for c = 1:n_c
        free = setdiff(1:n_m, idx(1:c-1), 'stable');
        [~, b] = max(lead(free, c)); idx(c) = free(b);
    end
    best = mean(first_click_prob(rate(idx, :), label, classes));
    for sweep = 1:6
        moved = false;
        for c = 1:n_c
            for m = setdiff(1:n_m, idx)
                trial = idx; trial(c) = m;
                a = mean(first_click_prob(rate(trial, :), label, classes));
                if a > best + 1e-12, best = a; idx = trial; moved = true; end
            end
        end
        if ~moved, break, end
    end
end
