function [clicks, kernel_class] = load_clicks(folder, n_kernels)
% LOAD_CLICKS  Recorded test clicks of the trained kernels, in seconds.
%   clicks{j, i}: click times of kernel j on test image i (s from the acquisition's start)
    clicks = {};
    kernel_class = zeros(1, n_kernels);
    for j = 1:n_kernels
        f = load(fullfile(folder, sprintf('clicks_k%d.mat', j)));
        for i = 1:numel(f.pat_ps)
            clicks{j, i} = double(f.pat_ps{i}) * 1e-12;
        end
        kernel_class(j) = f.kernel_class;
    end
end
