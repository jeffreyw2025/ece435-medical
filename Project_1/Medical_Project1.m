%% Jeffrey Wong | ECE-435 | Project #1- Imaging Fundamentals

% Function definitions are at the bottom of the file. Check them out- they
% contain some important context for how functions are called and some
% derivations!
clear
close all
clc

%% Notes on Scans

% CT Thorax 
% Pixel Spacing 0.703 mm 0028,0030;
xyspacing = 0.703; % Spacings used in Gaussian Blur
% Slice Thickness 1.25 mm 0018,0050; 
% Spacing Between Slides 0.625 mm 0018,0088; 
zspacing = 1.25 + 0.625; % Spacing in z impacted by both slice thickness and spacing
% Number of slices 237
num_slices = 237;

%% Part 1: Construction and Slicing

scan_size = size(rgb2gray(imread("thoraxCT\axial00001.jpg"))); 
volume = zeros([scan_size num_slices]); % Identifies size of scans and makes volume
for i = 1:num_slices
    % Hacky workaround to scan images, had to look up string formatting
    slice = rgb2gray(imread("thoraxCT\axial" + num2str(i, "%05d") + ".jpg"));
    % When passing in doubles value needs to be normalized between 0 and 1
    volume(:,:,i) = double(slice)/255;
end

% Configurable coordinates of slices
xloc = 289; 
yloc = 245; 
zloc = 162;

% Squeezing keeps stuff to two dimensions
xslice = squeeze(volume(xloc,:,:));
yslice = squeeze(volume(:,yloc,:));
zslice = squeeze(volume(:,:,zloc));

% x slice
figure
imshow(xslice)
axis on % Axis values have to be manually enabled for some reason
title("x = " + xloc)
xlabel("z (pixels)")
ylabel("x (pixels)")

% y slice
figure
imshow(yslice)
axis on
title("y = " + yloc)
xlabel("z (pixels)")
ylabel("y (pixels)")

% z slice
figure
imshow(zslice)
axis on
title("z = " + zloc)
xlabel("x (pixels)")
ylabel("y (pixels)")

%% Part 2: Sal y Pimienta

% Filtering on a "natural" slice

figure
subplot(1,2,1)
imshow(imfilter(xslice, ones(3)/9))
axis on
title("x = " + xloc + " slice w/ 3x3 mean filter")
subplot(1,2,2)
imshow(medfilt2(xslice, [3 3]))
axis on
title("x = " + xloc + " slice w/ 3x3 median filter")

figure
subplot(1,2,1)
imshow(imfilter(xslice, ones(5)/25))
axis on
title("x = " + xloc + " slice w/ 5x5 mean filter")
subplot(1,2,2)
imshow(medfilt2(xslice, [5 5]))
axis on
title("x = " + xloc + " slice w/ 5x5 median filter")

% Artificial Noise

salt_and_pepper_filtered(xslice, xloc, 0.03)
salt_and_pepper_filtered(xslice, xloc, 0.07)
salt_and_pepper_filtered(xslice, xloc, 0.15)

%% Part 3: Gauss' Wrath

% Gaussian Noise
% sigma_n (Noise stdev) = 0.07, sigma_f (Filter stdev) = 0.42
gaussian_filtered(xslice, xloc, 0.07, 0.42)
% sigma_n = 0.15, sigma_f = 0.9
gaussian_filtered(xslice, xloc, 0.15, 0.9)

% Gaussian Blur

% See the generate_Gaussian_kernel function for the derivation of how the
% kernel is generated

n = 5; % Affects kernel size
kersize = 2*n + 1;

% z image
% "Easy" because of 1:1 aspect ratio, the stdevs for x and y will be the
% same, resulting in isotropic Gaussian blur
% Now assume x/y/z are pixel offsets rather than distance offsets

dx = 2;
z_ker = generate_Gaussian_kernel(n, xyspacing, xyspacing, dx, dx);
figure
subplot(1,2,1)
imshow(zslice)
axis on
title("Original z = " + zloc + " slice")
subplot(1,2,2)
imshow(conv2(zslice, z_ker, "same"))
axis on
title("z = " + zloc + " slice w/ "+kersize+"x"+kersize+" Gaussian blur")

% x / y images
% Now our aspect ratio is not 1:1, so our Gaussian blur will be anisotripic

dz = 1;
xy_ker = generate_Gaussian_kernel(n, zspacing, xyspacing, dz, dx);

figure
subplot(1,2,1)
imshow(xslice)
axis on
title("Original x = " + xloc + " slice")
subplot(1,2,2)
imshow(conv2(xslice, xy_ker, "same"))
axis on
title("x = " + xloc + " slice w/ "+kersize+"x"+kersize+" Gaussian blur")

figure
subplot(1,2,1)
imshow(yslice)
axis on
title("Original y = " + yloc + " slice")
subplot(1,2,2)
imshow(conv2(yslice, xy_ker, "same"))
axis on
title("y = " + yloc + " slice w/ "+kersize+"x"+kersize+" Gaussian blur")

%% Part 4: Foramina

spine_slice = squeeze(volume(:,180,:));

% Edges of y-slice at y = 180
figure
subplot(1,2,1)
imshow(spine_slice)
title("Original y = 180 slice")
subplot(1,2,2)
imshow(edge(spine_slice, "canny"))
title("Edges for y = 180 slice")

% The vertebral disks can be seen in the lower third of the image as a
% series of large ovals, with the foramina above as a series of really
% small circles. Due to the noise in the center, there are quite a few
% spurious edges, with noise forming little "phantom" formamina and
% interfering with the disk, as well as adding some other extraneous edges
% above and below the sample that might be tied to other layers of the sampl

noisy_spine_slice = add_awgn(spine_slice, 0.07);
% Kernel for 3 pixel Gaussian filter
[P, Q] = meshgrid(-6:6, -6:6);
std3p_ker = exp(-P.^2/18 - Q.^2/18);
std3p_ker = std3p_ker/sum(std3p_ker, "all");

% Median + Part III Gaussian
figure
subplot(1,2,1)
imshow(edge(medfilt2(spine_slice, [3 3]), "canny"))
title("Edges for y = 180 slice w/ median filter")
subplot(1,2,2)
imshow(edge(conv2(spine_slice, z_ker, "same"), "canny"))
title("Edges for y = 180 slice w/ xz filter")

% AWGN + 3px Gaussian
figure
subplot(1,2,1)
imshow(edge(noisy_spine_slice, "canny"))
title("Edges for AWGN y = 180 slice")
subplot(1,2,2)
imshow(edge(conv2(spine_slice, std3p_ker, "same"), "canny"))
title("Edges for y = 180 slice w/ 3px filter")

% Bonus: Edges of z-slice at z = 42
figure
imshow(edge(zslice, "canny"))
title("Edges for z = 42 slice")

%% Function Definitions

% Adds salt + pepper noise to an entire image
function output = add_sp_noise(x, p)
    mask = rand(size(x)); % Generates random numbers to map to salt+pepper
    output = x;
    output(mask < p) = 0;
    output(mask > 1-p) = 1;
end

% Salt + Peppers image, filters using 3x3 mean and median filter, and
% displays results!
function salt_and_pepper_filtered(slice, xloc, p)
    slice_sp = add_sp_noise(slice, p);
    figure
    subplot(1,3,1)
    imshow(slice_sp)
    axis on
    title("S&P'd (p = "+p+") x = " + xloc + " slice")
    subplot(1,3,2)
    imshow(imfilter(slice_sp, ones(3)/9))
    axis on
    title("w/ 3x3 mean filter")
    subplot(1,3,3)
    imshow(medfilt2(slice_sp, [3 3]))
    axis on
    title("w/ 3x3 median filter")
end

% Adds additive Gaussian noise to an image
function output = add_awgn(x, sigma)
    n = sigma * randn(size(x));
    output = x + n;
end

% Adds AWGN of specified sigma_n and applies Gaussian filter of specified
% sigma_f, then displays results!
function gaussian_filtered(slice, xloc, sigma_n, sigma_f)
    slice_awgn = add_awgn(slice, sigma_n);
    figure
    subplot(1,3,1)
    imshow(slice)
    axis on
    title("Original x = " + xloc)
    subplot(1,3,2)
    imshow(slice_awgn)
    axis on
    title("x = " + xloc + " w/ AWGN (\sigma_n = "+sigma_n+")")
    subplot(1,3,3)
    imshow(imgaussfilt(slice_awgn, sigma_f))
    axis on
    title("Filtered (\sigma_f = "+sigma_f+") x = " + xloc)
end

% Function to generate a 2n+1 by 2n+1 Gaussian blur kernel
function ker = generate_Gaussian_kernel(n, dim1_spacing, dim2_spacing, dim1_fwhm, dim2_fwhm)
    % Suppose we have a FWHM of dw mm in the w axis. Then our standard
    % deviation, sigma_w, is given by the following:
    
    % 0.5 = exp(-(dw/2)^2/(2*sigma_w)^2)
    % -ln(0.5) = ln(2) = (dw/2)^2/(2*sigma_w)^2
    % sigma_w^2 = dw^2/(8ln2)
    % sigma_w = dw/sqrt(8ln2)
    % The 1D Gaussian becomes P(w) = 1/sqrt((pi/(4ln2))*dw^2) * exp(-(4w^2*ln2/dw^2))

    % The overall PSF is thus, for some constant alpha:
    % P(x,y,z) = alpha * exp(-(4x^2*ln2/dx^2)-(4y^2*ln2/dx^2)-(4z^2*ln2/dz^2))

    sigma_dim1 = dim1_fwhm/sqrt(8*log(2));
    sigma_dim2 = dim2_fwhm/sqrt(8*log(2));
    dim1_pts = (-n:n)*dim1_spacing;
    dim2_pts = (-n:n)*dim2_spacing;
    [dim1_grid, dim2_grid] = meshgrid(dim1_pts, dim2_pts);
    ker = exp(-(dim1_grid).^2/(2*sigma_dim1^2) - (dim2_grid).^2/(2*sigma_dim2^2));
    ker = ker./sum(ker, "all"); % Normalize kernel to 1
end