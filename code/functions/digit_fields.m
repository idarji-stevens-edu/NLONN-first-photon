function a = digit_fields(X, geo)
% DIGIT_FIELDS  Signal fields A exp(i pi v) of MNIST digits X (28 x 28 x n), Eq. (1).
    X = double(X); if max(X(:)) > 1, X = X / 255; end
    n = size(X, 3);
    a = complex(zeros(geo.G, geo.G, n));
    r = geo.off + (1:28 * geo.nsub);
    for k = 1:n
        ph = zeros(geo.G);
        ph(r, r) = kron(X(:, :, k), ones(geo.nsub));
        a(:, :, k) = geo.amp .* exp(1i * pi * ph);
    end
end
