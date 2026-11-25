function p = first_click_prob(rate, label, classes)
% FIRST_CLICK_PROB  Probability that the first click names the true class, Eq. (5).
%   rate(j, i): rate of kernel j (naming classes(j)) on image i; equal dwells
    rate = max(rate, 0);
    P = rate ./ max(sum(rate, 1), realmin);
    [~, row] = ismember(label, classes);
    p = P(sub2ind(size(P), row, 1:numel(label)));
end
