%% Jeffrey Wong | ECE-435 | Project #3- MRI Wavelet Compressed Sensing

clear; close all; clc;

%% Part I- Volume Navigation

i16_max = 2^15;
csi1_anatomical_T1 = double(niftiread(".\sub-CSI1_ses-16_run-01_T1w.nii\sub-CSI1_ses-16_run-01_T1w_defaced.nii"))/i16_max;
csi1_anatomical_T2 = double(niftiread(".\sub-CSI1_ses-16_T2w.nii\sub-CSI1_ses-16_T2w_defaced.nii"))/i16_max;
% Normalize and convert to float for consistency and to rein in FT magnitude

figure
subplot(2,3,1)
axis on
imagesc(squeeze(csi1_anatomical_T1(:,:,120)))
title("Axial slice 120, T1-weighted")
subplot(2,3,2)
axis on
imagesc(squeeze(csi1_anatomical_T1(:,169,:)))
title("Coronal slice 169, T1-weighted")
subplot(2,3,3)
axis on
imagesc(squeeze(csi1_anatomical_T1(87,:,:)))
title("Sagittal slice 87, T1-weighted")
subplot(2,3,4)
axis on
imagesc(squeeze(csi1_anatomical_T2(:,:,120)))
title("Axial slice 120, T2-weighted")
subplot(2,3,5)
axis on
imagesc(squeeze(csi1_anatomical_T2(:,169,:)))
title("Coronal slice 169, T2-weighted")
subplot(2,3,6)
axis on
imagesc(squeeze(csi1_anatomical_T2(87,:,:)))
title("Sagittal slice 87, T2-weighted")

% Normalize values so our MSEs are not freaking huge
guiding_image = squeeze(csi1_anatomical_T1(87,:,:));
figure
imagesc(guiding_image)
title("Guiding Image in Spatial Domain")
colorbar

%% Part III- Fourier Domain Compression?

guiding_image_FT = fftshift(fft2(guiding_image));
figure
imagesc(abs(guiding_image_FT))
title("Guiding Image in Fourier Domain")
colorbar

figure
histogram(abs(guiding_image_FT(:)), 1024, "Normalization","probability")
title("Fourier-Domain Guiding Image Magnitudes")

plot_fourier_compressed(guiding_image, 50)
plot_fourier_compressed(guiding_image, 75)
plot_fourier_compressed(guiding_image, 90)
plot_fourier_compressed(guiding_image, 98)

%% Part IV- Wavelet Domains

test_image = imread("cameraman.tif");
% Test Image 1- Cameraman
generate_wavelet_images(test_image, 'haar', 'Cameraman') % Haar or Daubiches 1
generate_wavelet_images(test_image, 'db4', 'Cameraman') % Daubiches 4
generate_wavelet_images(test_image, 'coif3', 'Cameraman') % Coiflet 3
% Test Image 2- Standard Basis Element
basis_element = zeros(256);
basis_element(128, 174) = 1;
generate_wavelet_images(basis_element, 'haar', 'Basis Element') 
generate_wavelet_images(basis_element, 'db4', 'Basis Element') 
generate_wavelet_images(basis_element, 'coif3', 'Basis Element')

plot_basis_FT(basis_element, 'haar');
plot_basis_FT(basis_element, 'db4');
plot_basis_FT(basis_element, 'coif3');

% Guiding Image!

% Wavelet decomposition of images
generate_wavelet_images(guiding_image, 'haar', 'Guiding Image') 
generate_wavelet_images(guiding_image, 'db4', 'Guiding Image') 
generate_wavelet_images(guiding_image, 'coif3', 'Guiding Image')

% Coefficient histograms
plot_wavelet_coeff_histogram(guiding_image, 'haar')
plot_wavelet_coeff_histogram(guiding_image, 'db4')
plot_wavelet_coeff_histogram(guiding_image, 'coif3')

% Out of these it's largely a tossup between db4 and coif3. I'll pick coif3
% at level 3 for reconstruction because there's a bit more spread 
% in the high coefficients.

[C, S] = wavedec2(guiding_image, 3, "coif3");
C = compress_lowest(C, 10);
reconstructed_coif_image = waverec2(C, S, "coif3");
mse = compute_mse(reconstructed_coif_image, guiding_image);
disp("Mean Squared Error of Reconstruction with 10% of coif3 coefficients dropped: "+ mse)
figure
subplot(1,2,1)
imagesc(guiding_image)
title("Original Image")
colorbar
subplot(1,2,2)
imagesc(reconstructed_coif_image)
title("Reconstructed Image with coif3 coefficients compressed, MSE = " + mse)
colorbar

%% Part V- How Sparse?

% Start with testing the guiding image at 1,2,3 levels
reconstruct_compressed_images(guiding_image, 1);
reconstruct_compressed_images(guiding_image, 2);
reconstruct_compressed_images(guiding_image, 3);

% Let's take a different slice of T1 (@ lv 2 for comparison)
reconstruct_compressed_images(squeeze(csi1_anatomical_T1(:,:,150)), 2);

% Try out a slice of T2
reconstruct_compressed_images(squeeze(csi1_anatomical_T2(100,:,:)), 2);

%% Part VI- Compressed Sensing

% I will limit myself to 2-level decompositions because 3-level takes too
% long (and doing 1000*n_iter 3-level decompositions will probaly brick my
% laptop), and of the 2-levels Daubiches 4 seems to fail the most gracefully.

guiding_image_FT = fftshift(fft2(guiding_image));
num_runs = 1000;
n_iter = 500;
eta = 0.01; % Step size (<= 0.01 seems reasonable)
lambda = 1e-3; % Detail L1 regularizer (<=1e-2 seems good)
disp("Performing PGD with step size = " + eta +" and regularization constant " + lambda)
for p = 70:-5:25
    run_fzs = zeros(num_runs, n_iter);
    final_images = zeros([size(guiding_image_FT), num_runs]);
    image_MSEs = zeros(1, num_runs);
    for i = 1:num_runs
        A = rand(size(guiding_image_FT)) < p/100;
        y_true = A .* guiding_image_FT;
        [final_images(:,:,i), ~, run_fzs(i, :)] = perform_PGD(y_true, A, n_iter, eta, lambda);
        image_MSEs(i) = compute_mse(final_images(:,:,i), guiding_image);
    end
    
    figure
    plot(1:n_iter,mean(run_fzs));
    title("Average objective function value of PGD")
    xlabel("Iteration")
    ylabel("f(z)")
    
    disp("MSE for p = " +p +"%: " + mean(image_MSEs))
    
    figure
    subplot(1,2,1)
    imagesc(guiding_image)
    title("Guiding Image")
    subplot(1,2,2)
    imagesc(final_images(:,:,1))
    title("Example reconstructed image from sparse sample matrix")
end
%% Function Definitions

% Part II
% Discards the lowest magnitude s percent of data points from a data array
function result = compress_lowest(data, s)
    data_sorted = sort(abs(data(:)), "ascend");
    target_pos = ceil(0.01 * s * length(data_sorted));
    target_val = data_sorted(target_pos);
    result = data .* (abs(data) > target_val);
end

% Computes mean-square error given an estimate array and "truth" array
function mse = compute_mse(y_est, y_truth)
    % Convert to double to avoid type issues
    mse = mean(abs(double(y_est) - double(y_truth)).^2, "all");
end

% Used in part III, converts raw data to Fourier domain, drops the lowest 
% magnitude s% of points from data then plots a reconstructed image from
% the Fourier domain coefficients and displays the associated MSE
function plot_fourier_compressed(data, s)
    data_FT = fftshift(fft2(data));
    compressed_data_FT = compress_lowest(data_FT, s);
    reconstructed_data = ifft2(fftshift(compressed_data_FT));
    mse = compute_mse(reconstructed_data, data);
    disp("Mean Squared Error of Reconstruction with "+s+"% of Fourier coefficients dropped: "+ mse)
    figure
    subplot(1,2,1)
    imagesc(data)
    title("Original Image")
    colorbar
    subplot(1,2,2)
    imagesc(reconstructed_data)
    title("Reconstructed Image with s = " + s + "%, MSE = " + mse)
    colorbar
end

% Plots wavelet transformed versions of data w/ specified wavelet domain
function generate_wavelet_images(data, dom_name, img_name)
    % Code shamelessly appropriated from https://www.mathworks.com/matlabcentral/answers/57296-displaying-the-image-of-wavedec2
    [C,S] = wavedec2(data,2,dom_name);
    A2 = appcoef2(C,S,dom_name,2); % 2-level approximation coefficients
    [H1,V1,D1] = detcoef2('all',C,S,1); % 1-level detail coefficients
    [H2,V2,D2] = detcoef2('all',C,S,2); % 2-level detail coefficients
    figure
    hold on
    imagesc([1 64], [256 193], A2);
    imagesc([65 128], [256 193], H2); 
    imagesc([1 64], [192 129], V2); 
    imagesc([65 128], [192 129], D2);
    imagesc([129 256], [256 129], H1); 
    imagesc([1 128], [128 1], V1); 
    imagesc([129 256], [128 1], D1);
    title("Wavelet Decomposition of "+ img_name + " using " + dom_name)
    xlim([1 256])
    ylim([1 256])
    colormap gray;
end

% Plots the Fourier Transform of the diagonal detail coefficients of the
% 1-level wavelet transform in a given basis to determine localization in
% frequency
function d_FT = plot_basis_FT(basis_ele, dom_name)
    [C,S] = wavedec2(basis_ele, 2, dom_name);
    d = detcoef2("d", C, S, 1);
    d_FT = fftshift(fft2(d));
    figure
    subplot(1,2,1)
    imagesc(d)
    title("Diagonal Detail Coefficients for " + dom_name)
    subplot(1,2,2)
    imagesc(abs(d_FT))
    title("Fourier Transform")
end

% Plots magintude of wavelet coefficients of a wavelet decomposition of an
% image
function plot_wavelet_coeff_histogram(data, dom_name)
    [C_1, ~] = wavedec2(data, 1, dom_name);
    [C_2, ~] = wavedec2(data, 2, dom_name);
    [C_3, ~] = wavedec2(data, 3, dom_name);
    figure
    subplot(1,3,1)
    histogram(abs(C_1(:)), "Normalization","probability")
    title("Wavelet coefficients for " +dom_name + " 1-level decomposition")
    subplot(1,3,2)
    histogram(abs(C_2(:)), "Normalization","probability")
    title("Wavelet coefficients for " +dom_name + " 2-level decomposition")
    subplot(1,3,3)
    histogram(abs(C_3(:)), "Normalization","probability")
    title("Wavelet coefficients for " +dom_name + " 3-level decomposition")
end

% Peforms wavelet decomposition and drops s = 0-99% of wavelet coefficients 
% in 1% increments. Plots the MSE s and plots sample images at 65%, 75%, 
% 85%, and 95%, then returns a matrix of MSEs and set of reconstructed
% images for each s in each basis
function [reconstruction_mses, haar_imgs, db4_imgs, coif3_imgs] = reconstruct_compressed_images(candidate_image, lv)
    img_set_size = [size(candidate_image), 99]; % We will generate a matrix of 99 images from each basis
    reconstruction_mses = zeros(3,100); % Each row will represent the MMSE 
    % of the reconstruction image using haar, db4, and coif3 bases
    haar_imgs = zeros(img_set_size);
    db4_imgs = zeros(img_set_size);
    coif3_imgs = zeros(img_set_size);
    for s = 1:99
        [C_haar, S_haar] = wavedec2(candidate_image, lv, 'haar');
        [C_db4, S_db4] = wavedec2(candidate_image, lv, 'db4');
        [C_coif3, S_coif3] = wavedec2(candidate_image, lv, 'coif3');
        C_haar = compress_lowest(C_haar, s);
        C_db4 = compress_lowest(C_db4, s);
        C_coif3 = compress_lowest(C_coif3, s);
        haar_imgs(:,:,s) = waverec2(C_haar, S_haar, 'haar');
        db4_imgs(:,:,s) = waverec2(C_db4, S_db4, 'db4');
        coif3_imgs(:,:,s) = waverec2(C_coif3, S_coif3, 'coif3');
        reconstruction_mses(1,s+1) = compute_mse(haar_imgs(:,:,s), candidate_image);
        reconstruction_mses(2,s+1) = compute_mse(db4_imgs(:,:,s), candidate_image);
        reconstruction_mses(3,s+1) = compute_mse(coif3_imgs(:,:,s), candidate_image);
    end
    
    % MSE Plot
    figure
    hold on
    plot(0:99, reconstruction_mses(1,:), "r--", 'DisplayName', "haar")
    plot(0:99, reconstruction_mses(2,:), "b--", 'DisplayName', "db4")
    plot(0:99, reconstruction_mses(3,:), "g--", 'DisplayName', "coif3")
    title('Reconstruction MMSEs using wavelet bases')
    xlabel("% Coefficients compresssed")
    ylabel("MMSE")
    yline(0.01, "k--", 'displayName', "Threshold") % Corresponds to 10% error
    legend
    
    figure
    subplot(2,2,1)
    imagesc(haar_imgs(:,:,65))
    title("65% of coefficients dropped (Haar)")
    subplot(2,2,2)
    imagesc(haar_imgs(:,:,75))
    title("75% of coefficients dropped (Haar)")
    subplot(2,2,3)
    imagesc(haar_imgs(:,:,85))
    title("85% of coefficients dropped (Haar)")
    subplot(2,2,4)
    imagesc(haar_imgs(:,:,95))
    title("95% of coefficients dropped (Haar)")
    
    figure
    subplot(2,2,1)
    imagesc(db4_imgs(:,:,65))
    title("65% of coefficients dropped (db4)")
    subplot(2,2,2)
    imagesc(db4_imgs(:,:,75))
    title("75% of coefficients dropped (db4)")
    subplot(2,2,3)
    imagesc(db4_imgs(:,:,85))
    title("85% of coefficients dropped (db4)")
    subplot(2,2,4)
    imagesc(db4_imgs(:,:,95))
    title("95% of coefficients dropped (db4)")
    
    figure
    subplot(2,2,1)
    imagesc(coif3_imgs(:,:,65))
    title("65% of coefficients dropped (coif3)")
    subplot(2,2,2)
    imagesc(coif3_imgs(:,:,75))
    title("75% of coefficients dropped (coif3)")
    subplot(2,2,3)
    imagesc(coif3_imgs(:,:,85))
    title("85% of coefficients dropped (coif3)")
    subplot(2,2,4)
    imagesc(coif3_imgs(:,:,95))
    title("95% of coefficients dropped (coif3)")
end

% Performs proximal gradient descent given a ground truth, a
% specified sensing matrix, a number of iterations, a step size eta, and a
% sparsity (L1) regularization parameter
function [final_image, MSE, f_z] = perform_PGD(y_true, A, n_iter, eta, lambda)
    % Initialize wavelet coefficients as all zeros (I don't love it but OK)
    [C_test, S] = wavedec2(y_true, 2, 'db4'); 
    % Note that S is consistent for all wavelet decompositions at the same
    % level, basis, and image size
    z = zeros(size(C_test));
    MSE = zeros(1,n_iter);
    f_z = zeros(1,n_iter);
    for i = 1:n_iter
        % For the first part of ISTA we just take the gradient on the part
        % of the objective function corresponding to fidelity
        z_intermediate = z - eta * MT_map(M_map(z, S, A) - y_true, A);
        % For the second part of ISTA we want to set each component to zero
        % if it is within eta * lambda of the origin, and descend towards
        % zero otherwise
        z = (abs(z_intermediate) - eta * lambda) .* (abs(z_intermediate) > eta * lambda) .* sign(z_intermediate);
        MSE(1,i) = compute_mse(M_map(z, S, A), y_true);
        f_z(1,i) = 0.5*sum(abs(M_map(z, S, A) - y_true).^2, "all") + lambda * sum(abs(z));
    end
    final_image = waverec2(z, S, 'db4');
end

% Maps from a set of wavelet coefficients Z (with bookkeeping matrix S) and
% sensing matrix A to a Fourier-domain image X (equivalent to multiplying
% by the M specified in the notes)
function X_kdom = M_map(Z, S, A)
    X_spatial = waverec2(Z, S, 'db4');
    X_kdom = fftshift(fft2(X_spatial));
    X_kdom = A .* X_kdom;

end

% Maps from a Fourier domain image X to a set of wavelet coefficients Z
% (equivalent to multiplying by the M^T specified in the notes)
function Z = MT_map(X_kdom, A)
    X_kdom = A .* X_kdom;
    X_spatial = real(ifft2(fftshift(X_kdom))); % Take real part to eliminate accretion of imaginary information
    [Z,~] = wavedec2(X_spatial, 2, 'db4');
end