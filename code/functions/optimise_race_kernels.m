function [V, history] = optimise_race_kernels(G, label, V0, max_iter)
% OPTIMISE_RACE_KERNELS  One kernel per class maximising the mean first-click probability, Eq. (5).
%   G: modes x images projections g(x); V0: modes x classes start. Gradient ascent with a
%   backtracking step, columns kept at unit norm. history: objective after each step.
    label = label(:)'; cls = unique(label);
    tgt = arrayfun(@(l) find(cls == l), label);
    V = V0 ./ vecnorm(V0);
    F = race_objective(V, G, tgt);
    t = 1; history = F;
    for it = 1:max_iter
        [~, Gr] = race_objective(V, G, tgt);
        Gr = Gr - V .* real(sum(conj(V) .* Gr, 1));          % keep columns unit norm
        if norm(Gr, 'fro') < 1e-14, break, end
        ok = false;
        for ls = 1:40
            Vn = V + t * Gr; Vn = Vn ./ vecnorm(Vn);
            Fn = race_objective(Vn, G, tgt);
            if Fn > F, ok = true; break, end
            t = t / 2;
        end
        if ~ok, break, end
        rel = (Fn - F) / max(F, eps);
        V = Vn; F = Fn; t = t * 1.5;
        history(end + 1) = F;
        if rel < 1e-10, break, end
    end
end

function [F, Gr] = race_objective(V, G, tgt)
% Mean probability that the first click names the true class, and its gradient in conj(V).
    A = V' * G; P = abs(A).^2; den = sum(P, 1); n = size(G, 2);
    lin = sub2ind(size(P), tgt, 1:n);
    num = P(lin);
    F = mean(num ./ den);
    if nargout < 2, return, end
    Wt = repmat(-(num ./ den.^2), size(P, 1), 1);
    Wt(lin) = Wt(lin) + 1 ./ den;
    Gr = (G * (conj(A) .* Wt).') / n;
end
