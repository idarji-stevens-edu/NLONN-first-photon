function fv = mask_validation(w, Zd, Yd, Zv, Yv)
% MASK_VALIDATION  Accuracy on the validation half with the rule from the design half.
    [~, S] = mask_accuracy(w, Zd, Yd); [~, d] = max(S, [], 2);
    p = w' * Zv; ip = zeros(size(p)); ip(p > 0) = 1 ./ p(p > 0);
    hit = Yv(d, :);                                   % does pixel k name image i's class
    fv = double(sum(sum((w .* Zv) .* hit, 1) .* ip)) / size(Zv, 2);
end
