function [first, trial_acc, n_trials] = first_click_race(clicks, kernel_class, label, tau, t_int)
% FIRST_CLICK_RACE  Decide each test image from its recorded clicks.
%   The kernels' acquisitions are merged on a common clock (time-division multiplexing).
%   first(i)     : class named by the first click of image i (single shot; NaN if no click)
%   trial_acc(i) : fraction of trials of length tau whose first click names the true class
%   n_trials(i)  : number of trials with at least one click
    [n_k, n_img] = size(clicks);
    n_seg = floor(t_int / tau);
    first = nan(1, n_img); trial_acc = nan(1, n_img); n_trials = zeros(1, n_img);
    for i = 1:n_img
        t_first = inf(n_k, 1);
        t_seg = inf(n_k, n_seg);              % first click of each kernel in each trial
        for j = 1:n_k
            t = sort(clicks{j, i});
            t = t(t >= 0 & t < t_int);
            if isempty(t), continue, end
            t_first(j) = t(1);
            seg = floor(t / tau) + 1;
            keep = seg <= n_seg; t = t(keep); seg = seg(keep);
            [s, k] = unique(seg, 'first');
            t_seg(j, s) = t(k) - (s - 1) * tau;
        end
        [tmin, jmin] = min(t_first);
        if isfinite(tmin), first(i) = kernel_class(jmin); end
        [tmin, jmin] = min(t_seg, [], 1);
        ok = isfinite(tmin);
        n_trials(i) = nnz(ok);
        if n_trials(i) > 0, trial_acc(i) = mean(kernel_class(jmin(ok)) == label(i)); end
    end
end
