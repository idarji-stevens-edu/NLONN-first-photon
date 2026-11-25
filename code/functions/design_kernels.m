function [C, w, dec, acc, history] = design_kernels(M, label, K, n_starts, seed, floor_level)
% DESIGN_KERNELS  Design K kernels and dwells on measured response matrices (Appendix B, Candidates).
%   M: d x d x n per-digit response matrices; label: classes 0-9
%   Starts: the single modes, the min-error measurement's rank-one parts, random superpositions.
%   Each start is optimised by fminunc on the soft accuracy (beta 30, then 100); the design with
%   the best hard training accuracy is returned.
%   history: [beta, hard accuracy, soft accuracy] after every iteration of the returned start
    d = size(M, 1);
    M = normalise_responses(M, floor_level);
    Y = double(label(:) == (0:9));
    rs = RandStream('mt19937ar', 'Seed', seed);

    % start 1: the single modes (plus random superpositions if K > d)
    Cr = complex(randn(rs, d, K - d), randn(rs, d, K - d)); Cr = Cr ./ vecnorm(Cr);
    starts = {{[eye(d), Cr], ones(K, 1)}};

    % start 2: the min-error measurement on the class-mean matrices, split into rank-one parts
    cl = find(any(Y, 1)); rho = zeros(d, d, numel(cl));
    for q = 1:numel(cl)
        Mc = mean(M(:, :, Y(:, cl(q)) > 0), 3); rho(:, :, q) = Mc / real(trace(Mc));
    end
    E = min_error_measurement(rho, sum(Y(:, cl), 1) / size(Y, 1));
    V = []; lam = [];
    for c = 1:size(E, 3)
        [U, e] = eig((E(:, :, c) + E(:, :, c)') / 2, 'vector'); k = e > 1e-9;
        V = [V, U(:, k)]; lam = [lam; e(k)];
    end
    [lam, o] = sort(lam, 'descend'); V = V(:, o);
    if size(V, 2) >= K
        starts{2} = {V(:, 1:K), lam(1:K)};
    else                                             % too few parts: pad with random superpositions
        Cr = complex(randn(rs, d, K - size(V, 2)), randn(rs, d, K - size(V, 2))); Cr = Cr ./ vecnorm(Cr);
        starts{2} = {[V, Cr], [lam; mean(lam) * ones(K - size(V, 2), 1)]};
    end

    % further starts: random superpositions
    while numel(starts) < n_starts
        Cr = complex(randn(rs, d, K), randn(rs, d, K)); Cr = Cr ./ vecnorm(Cr);
        starts{end + 1} = {Cr, ones(K, 1)};
    end

    % optimise each start
    acc = -inf;
    for s = 1:numel(starts)
        x = [real(starts{s}{1}(:)); imag(starts{s}{1}(:)); log(max(starts{s}{2}(:), 1e-6))];
        hist_s = zeros(0, 3);
        for beta = [30 100]
            record_iterate([], 'reset');
            o = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'Display', 'off', 'MaxIterations', 300, ...
                'MaxFunctionEvaluations', 2e5, 'OutputFcn', @(xx, ov, st) record_iterate(xx, st));
            x = fminunc(@(xx) -soft_accuracy(xx, d, K, M, Y, beta), x, o);
            X = record_iterate([], 'get');
            for k = 1:size(X, 2)
                [ah, as] = design_accuracy(X(:, k), d, K, M, Y, beta);
                hist_s(end + 1, :) = [beta, ah, as];
            end
        end
        [a, ~, dec_s] = design_accuracy(x, d, K, M, Y, 100);
        if a > acc
            acc = a; dec = dec_s; history = hist_s;
            [C, w] = unpack_kernels(x, d, K);
        end
    end
end

function a = soft_accuracy(x, d, K, M, Y, beta)
    [~, a] = design_accuracy(x, d, K, M, Y, beta);
end
