function E = min_error_measurement(rho, p)
% MIN_ERROR_MEASUREMENT  Minimum-error measurement of the states rho(:,:,c) with priors p (Helstrom).
%   Gradient ascent on E_c = X_c' X_c over isometries X, started from the pretty-good measurement.
    d = size(rho, 1); n_c = size(rho, 3);
    W = zeros(d, d, n_c);
    for c = 1:n_c, W(:, :, c) = p(c) * (rho(:, :, c) + rho(:, :, c)') / 2; end

    % restrict to the support of the total state
    S = sum(W, 3); [U, s] = eig((S + S') / 2, 'vector');
    V = U(:, s > 1e-10 * max(s)); r = size(V, 2);
    Wr = zeros(r, r, n_c);
    for c = 1:n_c
        A = V' * W(:, :, c) * V; Wr(:, :, c) = (A + A') / 2;
    end

    % pretty-good measurement as the start
    S = sum(Wr, 3); [U, s] = eig((S + S') / 2, 'vector');
    ok = s > eps(max(s)) * numel(s); q = zeros(size(s)); q(ok) = s(ok).^-0.5;
    S_half = U * diag(q) * U'; S_half = (S_half + S_half') / 2;
    X = zeros(n_c * r, r);
    for c = 1:n_c
        A = S_half * Wr(:, :, c) * S_half; [U, s] = eig((A + A') / 2, 'vector');
        ok = s > eps(max(s)) * numel(s); q = zeros(size(s)); q(ok) = sqrt(s(ok));
        B = U * diag(q) * U'; X((c-1)*r + (1:r), :) = (B + B') / 2;
    end
    [U, ~, Vs] = svd(X, 'econ'); X = U * Vs';
    F = 0;
    for c = 1:n_c, Xc = X((c-1)*r + (1:r), :); F = F + real(trace(Xc * Wr(:, :, c) * Xc')); end

    % gradient ascent on the isometries, with a backtracking step
    t = 1;
    for it = 1:500
        G = zeros(n_c * r, r);
        for c = 1:n_c, G((c-1)*r + (1:r), :) = X((c-1)*r + (1:r), :) * Wr(:, :, c); end
        H = X' * G; G = G - X * (H + H') / 2;            % tangent direction
        if norm(G, 'fro') < 1e-14, break, end
        accepted = false;
        for ls = 1:40
            [U, ~, Vs] = svd(X + t * G, 'econ'); Xn = U * Vs';
            Fn = 0;
            for c = 1:n_c, Xc = Xn((c-1)*r + (1:r), :); Fn = Fn + real(trace(Xc * Wr(:, :, c) * Xc')); end
            if Fn > F, accepted = true; break, end
            t = t / 2;
        end
        if ~accepted, break, end
        rel = (Fn - F) / max(1, abs(F));
        X = Xn; F = Fn; t = t * 1.5;
        if rel < 1e-12, break, end
    end
    E = zeros(d, d, n_c);
    for c = 1:n_c
        Xc = X((c-1)*r + (1:r), :); A = Xc' * Xc;
        E(:, :, c) = V * ((A + A') / 2) * V';
    end
end
