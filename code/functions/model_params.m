function p = model_params()
% MODEL_PARAMS  Geometry of the ideal NLONN model (Appendix B, Model of the ideal NLONN).
    p.N = 100;                       % grid points per side
    p.pitch = 1.092e-4;              % grid pitch (m), 10.92 mm window
    x = p.pitch * linspace(-p.N/2 + 1, p.N/2, p.N);
    [p.X, p.Y] = meshgrid(x, x);
    p.L = 5e-2;                      % crystal length (m)
    p.lens_f = 0.2;                  % L1, into the crystal (m)
    p.lens_f2 = 0.2;                 % L2, towards the fibre coupler (m)
    p.coupler_f = 11e-3;             % fibre-coupler asphere (m)
    p.fiber_mfd = 5e-6;              % fibre mode-field diameter (m)
    p.w_ill = 4.0e-3;                % pump illumination 1/e^2 radius (m)
    p.pump_w0 = 2e-3;                % waist of the HG masks written on the pump (m)
    p.signal_w0 = 2.83e-3;           % signal illumination 1/e^2 radius (m)
    p.lambda_p = 1558.53e-9;         % pump
    p.lambda_s = 1545.32e-9;         % signal
    p.lambda_f = p.lambda_s * p.lambda_p / (p.lambda_s + p.lambda_p);   % sum frequency
    p.n_p = 2.1345; p.n_s = 2.1349; p.n_f = 2.1747;   % crystal indices (pump, signal, sum frequency; Sellmeier fit, 40 C)
    p.delta_k = 66.3;                % phase mismatch (1/m) at which HG00 converts a blank signal best
    p.n_z = 401;                     % slices through the crystal
    p.u_halfwidth = 4e-3;            % half-width of the 1-D focal-plane grid (m)
    p.u_points = 4096;               % points on it
    p.basis_order = 11;              % HG modes m, n = 0 ... 10
end
