function y = hermite(n, x)
% HERMITE  Physicists' Hermite polynomial H_n(x).
    y = 0;
    for m = 0:floor(n/2)
        y = y + (((-1)^m) / (factorial(m) * factorial(n - 2*m)) .* ((2 .* x).^(n - 2*m)));
    end
    y = factorial(n) .* y;
end
