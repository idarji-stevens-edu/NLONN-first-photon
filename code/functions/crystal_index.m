function n = crystal_index(lambda, T)
% CRYSTAL_INDEX  Extraordinary index of 5 % MgO:PPLN at wavelength lambda (m) and temperature T (C) [Gayer2008].
    a = [5.756, 0.0983, 0.202, 189.32, 12.52, 1.32e-2];
    b = [2.86e-6, 4.7e-8, 6.113e-8, 1.516e-4];
    f = (T - 24.5) * (T + 570.82); L2 = (lambda * 1e6)^2;
    n = sqrt(a(1) + b(1) * f + (a(2) + b(2) * f) / (L2 - (a(3) + b(3) * f)^2) + (a(4) + b(4) * f) / (L2 - a(5)^2) - a(6) * L2);
end
