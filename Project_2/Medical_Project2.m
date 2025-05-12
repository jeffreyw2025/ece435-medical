%% Jeffrey Wong | ECE-435 | Project #2- OCT Image Extraction

clear; close all; clc;

%% Parameters and File Loading

N = 2048; % # pixels in camera
D_Bscan = 175; % # background images in B-scan
D_Mscan = 320; % # background images in M-scans

% Load files of scans and convert from lambda to k domain (Takes a while!)
load("L2K.mat")
bscan_ID = fopen("BScan_Layers.raw");
bscan = fread(bscan_ID, "uint16", "l"); % Files are 16-bit unsigned little-endian integers
bscan = reshape(bscan, N, []);
bscan = L2K * bscan; 
mscan1_ID = fopen("MScan1.raw");
mscan1 = fread(mscan1_ID, "uint16", "l"); 
mscan1 = reshape(mscan1, N, []);
mscan1 = L2K * mscan1;
mscan40_ID = fopen("MScan40.raw");
mscan40 = fread(mscan40_ID, "uint16", "l"); 
mscan40 = reshape(mscan40, N, []);
mscan40 = L2K * mscan40;
fclose("all"); % Don't want no memory leaks!

% Seperate scans into data and BG portions to pass into functions
bscan_bg = bscan(:,1:D_Bscan);
bscan_data = bscan(:,D_Bscan+1:end);
mscan1_bg = mscan1(:,1:D_Mscan);
mscan1_data = mscan1(:,D_Mscan+1:end);
mscan40_bg = mscan40(:,1:D_Mscan);
mscan40_data = mscan40(:,D_Mscan+1:end);

disp("Preprocessing complete!")

%% A-Scan

% Generate A-scan magnitudes
[a_scan_b1, bg_b1] = generate_ascan(bscan_data(:,420), bscan_bg, true);
a_scan_b1_mag = 20*log10(abs(a_scan_b1));
[a_scan_b2, ~] = generate_ascan(bscan_data(:,5311), bscan_bg);
a_scan_b2_mag = 20*log10(abs(a_scan_b2));
[a_scan_m1, ~] = generate_ascan(mscan1_data(:,9001), mscan1_bg);
a_scan_m1_mag = 20*log10(abs(a_scan_m1));
[a_scan_m40, ~] = generate_ascan(mscan40_data(:,9001), mscan40_bg);
a_scan_m40_mag = 20*log10(abs(a_scan_m40));

figure
subplot(2,2,1)
plot(-N/2:N/2-1,a_scan_b1_mag)
title("A-Scan Magnitude of B-Scan @ Layer 420")
ylabel("Magnitude (dB)")
xlim([-N/2 N/2-1])
subplot(2,2,2)
plot(-N/2:N/2-1,a_scan_b2_mag)
title("A-Scan Magnitude of B-Scan @ Layer 5311")
ylabel("Magnitude (dB)")
xlim([-N/2 N/2-1])
subplot(2,2,3)
plot(-N/2:N/2-1,a_scan_m1_mag)
title("A-Scan Magnitude of M-Scan1 @ Layer 9001")
ylabel("Magnitude (dB)")
xlim([-N/2 N/2-1])
subplot(2,2,4)
plot(-N/2:N/2-1,a_scan_m40_mag)
title("A-Scan Magnitude of M-Scan40 @ Layer 9001")
ylabel("Magnitude (dB)")
xlim([-N/2 N/2-1])

[a_scan_b1_nodeconv_nobgsub, ~] = generate_ascan(bscan_data(:,420), bscan_bg, false, false, false);
a_scan_b1_nodeconv_nobgsub_mag = 20*log10(abs(a_scan_b1_nodeconv_nobgsub));
[a_scan_b1_nodeconv, ~] = generate_ascan(bscan_data(:,420), bscan_bg, false, false, true);
a_scan_b1_nodeconv_mag = 20*log10(abs(a_scan_b1_nodeconv));

figure
subplot(3,1,1)
plot(-N/2:N/2-1,a_scan_b1_mag)
title("A-Scan Magnitude of B-Scan @ Layer 420 (Full Pipeline)")
ylabel("Magnitude (dB)")
xlim([-N/2 N/2-1])
subplot(3,1,2)
plot(-N/2:N/2-1,a_scan_b1_nodeconv_nobgsub_mag)
title("A-Scan Magnitude of B-Scan @ Layer 420 (No Deconv, No BGsub)")
ylabel("Magnitude (dB)")
xlim([-N/2 N/2-1])
subplot(3,1,3)
plot(-N/2:N/2-1,a_scan_b1_nodeconv_mag)
title("A-Scan Magnitude of B-Scan @ Layer 420 (No Deconv, w/BGsub)")
ylabel("Magnitude (dB)")
xlim([-N/2 N/2-1])

%% B-Scan

raw_bscan = generate_bscan(bscan_data, bscan_bg);
% Normalize data to 1 for imshow (imshow accepts floats from 0 to 1)
raw_bscan = raw_bscan/max(raw_bscan, [], "all");
figure
imshow(raw_bscan)
title("Raw B-Scan Image")
daspect([36 1 1])

gamma = 0.15;
processed_bscan = adjust_gamma(raw_bscan, gamma);
processed_bscan = processed_bscan(floor(N/2)+1:end,:);

figure
imshow(processed_bscan)
axis on
title("Processed B-Scan Image")
daspect([36 1 1])

% Perform horizontal edge detection to identify layers
sobel_bscan = edge(processed_bscan, "sobel", "horizontal");

figure
imshow(sobel_bscan)
axis on
title("Horizontal edges of B-Scan Image")
daspect([36 1 1])

nodeconv_bscan = generate_bscan(bscan_data, bscan_bg, false);
% Normalize data to 1 for imshow (imshow accepts floats from 0 to 1)
nodeconv_bscan = nodeconv_bscan/max(nodeconv_bscan, [], "all");
nodeconv_bscan = adjust_gamma(nodeconv_bscan, gamma);
nodeconv_bscan = nodeconv_bscan(floor(N/2)+1:end,:);

figure
imshow(nodeconv_bscan)
title("B-Scan Image without Deconvolution")
daspect([36 1 1])

%% M-Scan

% Sampling rate and timestep specifications
f_s = 97656.25;
del_t = 1/f_s;

[m1_scan, ~] = generate_ascan(mscan1_data, mscan1_bg, true);
m1_scan_mean = mean(abs(m1_scan),2);
figure
plot(1:N,20*log10(m1_scan_mean));
title("Average A-Scan Magnitude of M-Scan1")
xlabel("Pixel")
ylabel("Magnitude (dB)")
xlim([1 N])

[displacement_px1085_m1, ~] = perform_spdm(m1_scan, del_t, 1085, "1-tone M-Scan");
[displacement_px965_m1, ~] = perform_spdm(m1_scan, del_t, 965, "1-tone M-Scan");

[m40_scan, ~] = generate_ascan(mscan40_data, mscan40_bg, true);
m40_scan_mean = mean(abs(m40_scan),2);
figure
plot(1:N,20*log10(m40_scan_mean));
title("Average A-Scan Magnitude of M-Scan40")
xlabel("Pixel")
ylabel("Magnitude (dB)")
xlim([1 N])
[displacement_px1085_m40, freq_data_m40] = perform_spdm(m40_scan, del_t, 1085, "40-tone M-Scan");
[displacement_px965_m40, ~] = perform_spdm(m40_scan, del_t, 965, "40-tone M-Scan");

tone_freqs = extract_tones(freq_data_m40, f_s, 70000);

%% Function Definitions

% Generates a complex A-scan image or series of images from raw data
% Also returns (smoothed) background for reference
function [scan, bg] = generate_ascan(data, bg, verbose, enable_deconv, enable_background_sub)
    % Use arguments block to enable/disable certain features for report
    arguments
        data
        bg
        verbose = false % Set to true to enable logging of times and track processes
        enable_deconv = true % Set to false to skip deconvolution
        enable_background_sub = true % Set to false to skip background subtraction
    end
    N = size(data,1); % Need # of points in scan for polyfit and window size

    % Background Subtraction / Fitting
    if(verbose)
        disp("Performing (1) Background Subtraction + Fitting")
        tic
    end
    bg = mean(bg, 2); % Average out background
    if(enable_background_sub)
        data = data - bg; % Subtract off the raw average bg only from data
    end
    % polyval(polyfit()) returns a row vector, want to transpose to form col vector
    bg = (polyval(polyfit(1:N, bg, 15), 1:N)).';
    if(verbose)
        toc
    end

    % Windowing
    if(verbose)
        disp("Performing (2) Windowing")
        tic
    end
    hwind = hamming(N); % Creates window based on # of data rows (should be 2048)
    data = data .* hwind; % Window by multiplying in k domain, hwind is a col vector so should broadcast automatically
    if(verbose)
        toc
    end

    % Deconvolution
    if(enable_deconv)
        if(verbose)
            disp("Performing (3) Deconvolution")
            tic
        end
        data = data./bg;
        if(verbose)
            toc
        end
    end
    % FFT + Averaging
    if(verbose)
        disp("Performing (4) FFT")
        tic
    end
        scan = fftshift(fft(data),1);
    if(verbose)
       toc
    end
end

% The B-scan function is basically the A-scan function without averaging,
% so we can just do the BG processing once!
function img = generate_bscan(data, bg, enable_deconv)
    arguments
        data
        bg
        enable_deconv = true
    end
    N = size(data,1);

    bg = mean(bg, 2); % Average out background
    data = data - bg; % Subtract off the raw average bg only from data
    bg = (polyval(polyfit(1:N, bg, 15), 1:N)).';

    hwind = hamming(N);
    data = data .* hwind;
    if(enable_deconv)
        data = data./bg;
    end
    img = abs(fftshift(fft(data),1));
end

% Takes in a 0-1 valued image and performs gamma correction
function img = adjust_gamma(data, gamma)
    img = data.^gamma;
end

function [tdom_data, freq_data] = perform_spdm(data, del_t, pixel_pos, scan_name)
    f_s = 1/del_t; % Used to plot frequency data
    data = data(pixel_pos, :);
    tdom_data = unwrap(angle(data)) * (1300 / (4*pi)); % Use SPDM equation w/ lambda_0 = 1300 nm and n = 1
    % Use Chebyshev II filter to eliminate near-DC components without
    % attenuating anything else (mostly)
    [b,a] = cheby2(5, 30, 500/(f_s/2), "high");
    filtered_tdom_data = filter(b, a, tdom_data);
    % filtered_tdom_data = data - mean(data);
    freq_data = fftshift(fft(filtered_tdom_data));

    figure
    subplot(2,1,1)
    plot(1000 * del_t .* (1:length(tdom_data)), tdom_data)
    title("SPDM Time Domain Data for "+scan_name+" @ pixel " + pixel_pos)
    xlabel("Time (ms)")
    ylabel("Displacement (nm)")
    subplot(2,1,2)
    plot(linspace(-f_s/2, f_s/2, length(freq_data)), abs(freq_data))
    title("SPDM Frequency Domain Data for "+scan_name+" @ pixel " + pixel_pos)
    xlabel("Frequency (Hz)")
    ylabel("Displacement (nm)")
end

% Extracts frequency components (in kHz) from data at a certain f_s that exceeds a
% specified threshold (Used to extract tones)
function tone_freqs = extract_tones(data, f_s, threshold)
    frequencies = linspace(-f_s/2, f_s/2, length(data));
    tone_freqs = frequencies(data > threshold & frequencies > 0);
    tone_freqs = tone_freqs / 1000;
end