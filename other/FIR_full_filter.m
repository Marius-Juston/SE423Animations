% -------------------------------------------------------------------------
% Advanced Filter Visualization Framework
% -------------------------------------------------------------------------

% Clean environment
clear; clc; close all;

% 1. Sampling & Signal Setup
fs = 2;                 % Nyquist = 1.0 (aligns with MATLAB fir1 bounds)
N = 2048 * 1;
t = (0:N-1)/fs;

% Multi-tone test signal (Frequencies mapped safely below Nyquist = 1.0)
f_comps = [0.1, 0.25, 0.50, 0.80]; 
A = [1, 1, 1, 1];

x = zeros(size(t));
for k = 1:length(f_comps)
    x = x + A(k)*sin(2*pi*f_comps(k)*t);
end

% 2. Filter Design 
% Utilizing fir1 - Note: Normalized frequency 1.0 is Nyquist (pi rad/sample)

% 21st Order Low-Pass Filter
Order_low = 21; 
Wn_low = 0.2; 
b_low = fir1(Order_low, Wn_low, 'low');

% 22nd Order High-Pass Filter (Must be even order for high-pass to preserve Nyquist)
Order_high = 22; 
Wn_high = 0.3; 
b_high = fir1(Order_high, Wn_high, 'high');

% 21st Order Band-Pass Filter
Order_band = 21; 
Wn_band = [0.35 0.65]; 
b_band = fir1(Order_band, Wn_band, 'bandpass');

% 3. Signal Filtering (Using zero-phase filtfilt to prevent group delay)
y_low  = filter(b_low, 1, x);
y_high = filter(b_high, 1, x);
y_band = filter(b_band, 1, x);

% 4. Master Visualization Layout
fig = figure('Position', [100, 100, 1200, 1000]);
tiledlayout(4, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% ----- Row 1: Input -----
nexttile; 
plot(t, x, 'k', 'LineWidth', 1); 
title('Time Domain: Input Signal'); xlabel('Time (s)'); ylabel('Amplitude');
xlim([0 150]); grid on;

nexttile; 
plotSpectrumStem(x, fs, [], 'none', 'Frequency Domain: Input Spectrum');

% ----- Row 2: Low-Pass -----
nexttile; 
plot(t, y_low, 'b', 'LineWidth', 1); 
title('Time Domain: Low-pass Output'); xlabel('Time (s)'); ylabel('Amplitude');
xlim([0 150]); grid on;

nexttile; 
plotSpectrumStem(y_low, fs, Wn_low, 'low', 'Frequency Domain: Low-pass Spectrum');

% ----- Row 3: High-Pass -----
nexttile; 
plot(t, y_high, 'r', 'LineWidth', 1); 
title('Time Domain: High-pass Output'); xlabel('Time (s)'); ylabel('Amplitude');
xlim([0 150]); grid on;

nexttile; 
plotSpectrumStem(y_high, fs, Wn_high, 'high', 'Frequency Domain: High-pass Spectrum');

% ----- Row 4: Band-Pass -----
nexttile; 
plot(t, y_band, 'g' , 'LineWidth', 1); 
title('Time Domain: Band-pass Output'); xlabel('Time (s)'); ylabel('Amplitude');
xlim([0 150]); grid on;

nexttile; 
plotSpectrumStem(y_band, fs, Wn_band, 'band', 'Frequency Domain: Band-pass Spectrum');

exportTransparentSVG(fig, 'filter_analysis_full.svg');

% =========================================================================
% Helper Functions
% =========================================================================

function plotSpectrumStem(sig, fs, cutoff, filterType, titleStr)
    N = length(sig);
    % Calculate single-sided magnitude properly scaled by 2/N
    X = abs(fft(sig))/(N/2); 
    f = (0:N/2) * (fs/N);

    hold on;
    % Contextual visual shading of the passband
    if ~strcmp(filterType, 'none')
        y_lims = [0 1.2];
        patch_color = [0.85 0.85 0.85];
        
        if strcmp(filterType, 'low')
            patch([0 cutoff cutoff 0], [y_lims(1) y_lims(1) y_lims(2) y_lims(2)], ...
                  patch_color, 'EdgeColor', 'none', 'FaceAlpha', 0.6);
        elseif strcmp(filterType, 'high')
            patch([cutoff 1.0 1.0 cutoff], [y_lims(1) y_lims(1) y_lims(2) y_lims(2)], ...
                  patch_color, 'EdgeColor', 'none', 'FaceAlpha', 0.6);
        elseif strcmp(filterType, 'band')
            patch([cutoff(1) cutoff(2) cutoff(2) cutoff(1)], [y_lims(1) y_lims(1) y_lims(2) y_lims(2)], ...
                  patch_color, 'EdgeColor', 'none', 'FaceAlpha', 0.6);
        end
    end
    
    % Proper discrete stem plot
    stem(f, X(1:N/2+1), 'filled', 'MarkerSize', 5, 'Color', 'k', 'LineWidth', 1.2);
    
    title(titleStr);
    xlabel('Normalized Frequency (\times\pi rad/sample)');
    ylabel('Magnitude');
    xlim([0 1.0]);
    ylim([0 1.2]);
    grid on;
    hold off;
end

function exportTransparentSVG(fig, filename)
    set(fig, 'Color', 'none');
    axes = findall(fig, 'type', 'axes');
    set(axes, 'Color', 'none');
    set(fig, 'InvertHardcopy', 'off');  
    print(fig, filename, '-dsvg');
end