%% Run every analysis script in order (about 2.5 h; most of it the two phase layers and the trained mask)
% Each script prints the numbers it reproduces and writes its CSV files to ../results.
cd(fileparts(mfilename('fullpath')));      % each script also clears the workspace

%% Bench data (Sec. IV A, V; Appendix C)
run('three_class_order3.m');
run('accuracy_vs_mode_order.m');
run('ten_class_order2.m');
run('reference_drift_flags.m');

%% Tomography and kernel design (Appendix B)
run('tomography_and_design.m');

%% Linear optics and cameras (Appendix A)
run('pixel_bound.m');
run('lens_camera_and_fixed_filters.m');
run('object_plane_phase_layer.m');
run('fourier_plane_phase_layer.m');
run('masked_cameras.m');

%% Ideal NLONN, comparison, subset choice, Fig. 1 data
run('crystal_model.m');
run('ideal_nlonn.m');
run('photon_number.m');
run('comparison_margins.m');
run('class_subset_choice.m');
run('network_figure_data.m');
