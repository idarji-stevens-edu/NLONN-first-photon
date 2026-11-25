function [f_add, u_add, f_rem, u_rem] = mask_best_move(A, Z, Y)
% MASK_BEST_MOVE  Accuracy after the best single-pixel addition to, and removal from, the set A.
%   All candidates are evaluated at once.
    n = size(Z, 2); n_c = size(Y, 1); a = find(A); p = sum(Z(a, :), 1);
    % additions
    U = find(~A & any(Z > 0, 2));
    Sa = zeros(numel(a), numel(U), n_c); su = zeros(numel(U), n_c);
    for c = 1:n_c
        ii = Y(c, :); W = 1 ./ (p(ii) + Z(U, ii));
        Sa(:, :, c) = Z(a, ii) * W'; su(:, c) = sum(Z(U, ii) .* W, 2);
    end
    f = double(reshape(sum(max(Sa, [], 3), 1), [], 1) + max(su, [], 2)) / n;
    [f_add, j] = max(f); u_add = U(j);
    % removals
    if numel(a) <= 1, f_rem = -Inf; u_rem = []; return, end
    S = zeros(numel(a), numel(a), n_c);
    for c = 1:n_c
        ii = Y(c, :); pr = p(ii) - Z(a, ii);
        W = zeros(size(pr)); W(pr > 1e-15) = 1 ./ pr(pr > 1e-15);
        S(:, :, c) = Z(a, ii) * W';                   % kept pixel x removed pixel
    end
    M = max(S, [], 3); M(logical(eye(numel(a)))) = 0;
    f = double(sum(M, 1)) / n; [f_rem, j] = max(f); u_rem = a(j);
end
