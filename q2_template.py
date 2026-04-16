import cv2
import numpy as np
import time
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt

# ==============================================================================
# 0. Setup and Image Loading
# ==============================================================================
print("--- 0. Setup: Loading Images ---")

'''
TODO: Load the original image 'bonn.jpg' and noisy image 'bonn_noisy.jpg'
Convert both to grayscale and prepare the noisy image in float format (0-1 range)
Calculate and print the PSNR of the noisy image compared to the original
'''

# Load images here
original_img_color = cv2.imread("bonn.jpg")  # Load bonn.jpg
original_img_gray = cv2.cvtColor(original_img_color, cv2.COLOR_BGR2GRAY)   # Convert to grayscale
noisy_img = cv2.cvtColor(cv2.imread("bonn_noisy.jpg"), cv2.COLOR_BGR2GRAY)      # Load bonn_noisy.jpg and convert to grayscale
noisy_img_float_01 = noisy_img.astype(np.float32) / 255.0  # Convert noisy image to float format (0-1)

# Calculate PSNR of noisy image
psnr_noisy = peak_signal_noise_ratio(original_img_gray, noisy_img, data_range=255)

# Display original and noisy images
# TODO: Create a figure showing original and noisy images side by side
cv2.imshow("Original vs. Noisy", cv2.hconcat((original_img_gray, noisy_img)))
cv2.waitKey(0)
cv2.destroyAllWindows()

# ==============================================================================
# Custom Filter Definitions (for parts a, b, c)
# ==============================================================================

def custom_gaussian_filter(image, kernel_size, sigma):
    """
    Custom Gaussian Filter - Implement convolution from scratch
    
    Args:
        image: Input image (float, 0-1 range)
        kernel_size: Size of the Gaussian kernel (odd integer)
        sigma: Standard deviation of the Gaussian
    
    Returns:
        Filtered image (float, 0-1 range)
    
    TODO: 
    1. Create Gaussian kernel using the formula: G(x,y) = exp(-(x^2 + y^2)/(2*sigma^2))
    2. Normalize the kernel so it sums to 1
    3. Pad the image using reflect mode
    4. Apply convolution manually using nested loops
    """
    image = np.array(image, dtype=np.float64)
    # Get a kernel row-vector with symmetric values
    
    assert kernel_size % 2 == 1

    k = kernel_size // 2
    x = np.arange(-k, k+1, 1)
    y = np.arange(-k, k+1, 1)
    # Build Marix 
    kernel_x, kernel_y = np.meshgrid(x,  y)
    # Calculate the kernel values and normalize
    kernel_matrix = np.exp(-(kernel_x**2 + kernel_y**2)/(2*sigma**2), dtype=np.float64)
    kernel_matrix = kernel_matrix / np.sum(kernel_matrix)
    # Reflect pad with np function 
    image_pad = np.pad(image, pad_width=kernel_size//2, mode="reflect")

    # For every pixel 
    filtered = np.zeros_like(image)

    for i in range(image.shape[0]): 
        for j in range(image.shape[1]): 
            img_reg = image_pad[i:i+kernel_size, j:j+kernel_size]
            # Due to symmetric property of Gaussian no need to rotate the kenel
            filtered[i,j] = np.sum(img_reg*kernel_matrix)

    return filtered

def custom_median_filter(image, kernel_size):
    """
    Custom Median Filter - Implement median calculation from scratch
    
    Args:
        image: Input image (float, 0-1 range)
        kernel_size: Size of the median filter window (odd integer)
    
    Returns:
        Filtered image (float, 0-1 range)
    
    TODO:
    1. Pad the image using reflect mode
    2. For each pixel, extract the neighborhood window
    3. Calculate the median of the window
    4. Assign the median value to the output pixel
    """
    # Get the image and pad with reflect
    image = np.array(image, dtype=np.float64)
    image_pad = np.pad(array=image, pad_width=kernel_size//2, mode="reflect")

    # Get neighborhood
    assert kernel_size % 2 == 1

    filtered = np.zeros_like(image)

    for i in range(image.shape[0]): 
        for j in range(image.shape[1]): 
            matrix_window = image_pad[i:i+kernel_size, j:j+kernel_size]
            flattened_array = matrix_window.flatten()
            median_value = np.median(flattened_array)
            filtered[i][j] = median_value

    return filtered


def custom_bilateral_filter(image, d, sigma_color, sigma_space):
    """
    Custom Bilateral Filter
    
    Args:
        image: Input image (float, 0-1 range)
        d: Diameter of the pixel neighborhood
        sigma_color: Filter sigma in the color space (0-1 range for float images)
        sigma_space: Filter sigma in the coordinate space
    
    Returns:
        Filtered image (float, 0-1 range)
    
    TODO:
    1. Pad the image
    2. For each pixel:
       a. Calculate spatial weights based on distance from center
       b. Calculate range weights based on intensity difference
       c. Combine weights and compute weighted average
    3. Normalize by sum of weights
    """
    image = np.array(image, dtype=np.float64)
    pad = d // 2
    image_pad = np.pad(array=image, pad_width=pad, mode="reflect")

    filtered = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]): 
            region = image_pad[i:i+d, j:j+d]
            weights = np.zeros_like(region)
            # k,l
            for k in range(d): 
                for l in range(d):
                    dist_sq = (k - pad)**2 + (l - pad)**2
                    g_s = np.exp(-dist_sq / (2 * sigma_space**2))
                    
                    diff_sq = (region[k, l] - image[i, j])**2
                    f_r = np.exp(-diff_sq / (2 * sigma_color**2))

                    # Due to exp rules to get w(i,j,k,l) just multiply
                    weights[k,l] = g_s * f_r
            
            filtered[i,j] = np.sum(region * weights) / np.sum(weights)
    
    return filtered

            

# ==============================================================================
# 1. Filter Application (Parts a, b, c)
# ==============================================================================
print("\n--- 1. Filter Application (Parts a, b, c) ---")

# Default Parameters
K_DEFAULT = 7
S_DEFAULT = 2.0
D_DEFAULT = 9
SC_DEFAULT = 100  # cv2 range (0-255)
SS_DEFAULT = 75 

# Matplot lib show func
def show_images(title, img1, img2, img3, labels=["Noisy","custom","cv2"]):
    plt.figure(figsize=(15,5))
    for i, img in enumerate([img1, img2, img3]):
        plt.subplot(1,3,i+1)
        if img.ndim == 2:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
        plt.title(f"{labels[i]}")
        plt.axis("off")
    plt.suptitle(title)
    plt.show()

# -------------------------- a) Gaussian Filter --------------------------
print("a) Applying Gaussian Filter...")
'''
TODO: 
1. Apply Gaussian filter using cv2.GaussianBlur()
2. Apply your custom Gaussian filter
3. Calculate PSNR for both results
4. Display the results in a figure with 3 subplots (noisy, cv2 result, custom result)
'''

denoised_gaussian_cv2 = cv2.GaussianBlur(noisy_img, (K_DEFAULT,K_DEFAULT), S_DEFAULT)
psnr_gaussian_cv2 = peak_signal_noise_ratio(original_img_gray, denoised_gaussian_cv2)

denoised_gaussian_custom = custom_gaussian_filter(noisy_img_float_01, K_DEFAULT, S_DEFAULT)
psnr_gaussian_custom = peak_signal_noise_ratio(original_img_gray, (denoised_gaussian_custom*255).astype(np.uint8))

# Display results here
show_images("Gaussian Filter comparison", noisy_img, denoised_gaussian_cv2, (denoised_gaussian_custom*255).astype(np.uint8), 
            labels=["Noisy", "CV2 Gaussian", "Custom Gaussian"])


# -------------------------- b) Median Filter --------------------------
print("b) Applying Median Filter...")
'''
TODO:
1. Apply Median filter using cv2.medianBlur()
2. Apply your custom Median filter
3. Calculate PSNR for both results
4. Display the results in a figure with 3 subplots
'''

denoised_median_cv2 = cv2.medianBlur(noisy_img, K_DEFAULT)
psnr_median_cv2 = peak_signal_noise_ratio(original_img_gray, denoised_median_cv2)

denoised_median_custom = custom_median_filter(noisy_img_float_01, K_DEFAULT)
psnr_median_custom = peak_signal_noise_ratio(original_img_gray, (denoised_median_custom*255).astype(np.uint8))


# Display results here
show_images("Median Filter comparison", noisy_img, denoised_median_cv2, (denoised_median_custom*255).astype(np.uint8), 
            labels=["Noisy", "CV2 Median", "Custom Median"])

# -------------------------- c) Bilateral Filter --------------------------
print("c) Applying Bilateral Filter...")
'''
TODO:
1. Apply Bilateral filter using cv2.bilateralFilter()
2. Apply your custom Bilateral filter (remember to scale sigma_color for 0-1 range)
3. Calculate PSNR for both results
4. Display the results in a figure with 3 subplots
'''

denoised_bilateral_cv2 = cv2.bilateralFilter(noisy_img, D_DEFAULT, SC_DEFAULT, SS_DEFAULT)
psnr_bilateral_cv2 = peak_signal_noise_ratio(original_img_gray, denoised_bilateral_cv2)

denoised_bilateral_custom = custom_bilateral_filter(noisy_img_float_01, D_DEFAULT, SC_DEFAULT, SS_DEFAULT)
psnr_bilateral_custom = peak_signal_noise_ratio(original_img_gray, (denoised_bilateral_custom*255).astype(np.uint8))

# Display results here
show_images("Biliteral Filter comparison", noisy_img, denoised_bilateral_cv2, (denoised_bilateral_custom*255).astype(np.uint8), 
            labels=["Noisy", "CV2 Biliteral", "Custom Biliteral"])

# ==============================================================================
# 2. Performance Comparison (Part d)
# ==============================================================================
print("\n--- d) Performance Comparison ---")
'''
TODO:
1. Compare PSNR values of all three filters
2. Determine which filter performs best
3. Display side-by-side comparison of all filtered images
4. Print the results with the best performing filter highlighted
'''
psnr_values = {
    "Gaussian CV2": psnr_gaussian_cv2,
    "Gaussian Custom": psnr_gaussian_custom,
    "Median CV2": psnr_median_cv2,
    "Median Custom": psnr_median_custom,
    "Bilateral CV2": psnr_bilateral_cv2,
    "Bilateral Custom": psnr_bilateral_custom
}
# Get best performing filter and print it 
best_filter_name = max(psnr_values, key=psnr_values.get)
best_psnr = psnr_values[best_filter_name]

print("\n=== PSNR Values ===")
for name, value in psnr_values.items():
    if name == best_filter_name:
        print(f"  {name}: {value:.2f} dB (BEST)")
    else:
        print(f"  {name}: {value:.2f} dB")

# Loop 
plt.figure(figsize=(18,6))
filtered_images = [
    (denoised_gaussian_cv2, "Gaussian CV2"),
    ((denoised_gaussian_custom*255).astype(np.uint8), "Gaussian Custom"),
    (denoised_median_cv2, "Median CV2"),
    ((denoised_median_custom*255).astype(np.uint8), "Median Custom"),
    (denoised_bilateral_cv2, "Bilateral CV2"),
    ((denoised_bilateral_custom*255).astype(np.uint8), "Bilateral Custom")
]
for i, (img, title) in enumerate(filtered_images): 
    plt.subplot(2,3,i+1)
    if img.ndim == 2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")

plt.suptitle("All Filtered Images Comparison")
plt.show()


# ==============================================================================
# 3. Parameter Optimization (Part e)
# ==============================================================================

def run_optimization(original_img, noisy_img):
    """
    Optimize parameters for all three filters to maximize PSNR
    
    Args:
        original_img: Original clean image
        noisy_img: Noisy image to be filtered
    
    Returns:
        Dictionary containing optimal parameters and best PSNR for each filter
    
    TODO:
    1. For Gaussian filter: iterate over kernel_sizes and sigma values
    2. For Median filter: iterate over kernel_sizes
    3. For Bilateral filter: iterate over d, sigma_color, and sigma_space values
    4. Track the best PSNR and corresponding parameters for each filter
    5. Return results as a dictionary
    
        """
    
    # !!!! This will take ages
    gaussian_kernels = [3, 5, 7, 9]
    gaussian_sigmas = [0.5, 1.0, 1.5, 2.0]

    median_kernels = [3, 5, 7, 9]

    bilateral_diameters = [7, 9, 11]
    bilateral_sigma_colors = [0.05, 0.1, 0.2]  # for 0–1 images
    bilateral_sigma_spaces = [1, 2, 3]
    

    # Smaller values to just check the implementation
    """ 
    gaussian_kernels = [5]       
    gaussian_sigmas = [1.0]      
    median_kernels = [5 ]       
    bilateral_diameters = [9]       
    bilateral_sigma_colors = [0.1]  
    bilateral_sigma_spaces = [2]    
    """
    results = {}

    best_psnr = -np.inf
    best_params = None
    for k_size in gaussian_kernels:
        for sigma_val in gaussian_sigmas:
            filtered = custom_gaussian_filter(noisy_img, k_size, sigma_val)
            psnr_val = peak_signal_noise_ratio(original_img, (filtered * 255).astype(np.uint8))
            if psnr_val > best_psnr:
                best_psnr = psnr_val
                best_params = (k_size, sigma_val)
    results["Gaussian Filter"] = {"Kernel Size": best_params[0],"Sigma": best_params[1],"PSNR": best_psnr}

    best_psnr = -np.inf
    best_params = None
    for k_size in median_kernels:
        filtered = custom_median_filter(noisy_img, k_size)
        psnr_val = peak_signal_noise_ratio(original_img, (filtered * 255).astype(np.uint8))
        if psnr_val > best_psnr:
            best_psnr = psnr_val
            best_params = k_size
    results["Median Filter"] = { "Kernel Size": best_params, "PSNR": best_psnr}

    # This will take ages
    best_psnr = -np.inf
    best_params = None
    for d in bilateral_diameters:
        for sc in bilateral_sigma_colors:
            for sp in bilateral_sigma_spaces:
                filtered = custom_bilateral_filter(noisy_img, d, sc, sp)
                psnr_val = peak_signal_noise_ratio(original_img, (filtered * 255).astype(np.uint8))
                if psnr_val > best_psnr:
                    best_psnr = psnr_val
                    best_params = (d, sc, sp)
    results["Bilateral Filter"] = {"Diameter": best_params[0],"Sigma Color": best_params[1],"Sigma Space": best_params[2],"PSNR": best_psnr}

    return results



'''
TODO:
1. Call run_optimization() function
2. Extract optimal parameters for each filter
3. Apply filters using optimal parameters
4. Display the optimized results in a 2x2 grid (noisy + 3 optimal filters)
5. Print the optimal parameters clearly
'''
opt_results = run_optimization(original_img_gray, noisy_img_float_01)
print("=== Optimal Parameters ===")
for name, vals in opt_results.items(): 
    print(f"\nFor {name} the following params are optimal:")
    for k,v in vals.items(): 
        print(f"{k}:{v}")

gauss_opt = custom_gaussian_filter(noisy_img_float_01, opt_results["Gaussian Filter"]["Kernel Size"], 
                                   opt_results["Gaussian Filter"]["Sigma"])

median_opt = custom_median_filter(noisy_img_float_01, opt_results["Median Filter"]["Kernel Size"])

bilateral_opt = custom_bilateral_filter(noisy_img_float_01, opt_results["Bilateral Filter"]["Diameter"], 
                                       opt_results["Bilateral Filter"]["Sigma Color"],opt_results["Bilateral Filter"]["Sigma Space"])

plt.figure(figsize=(12,10))
for i, (img, title) in enumerate([
    (noisy_img, "Noisy"),
    ((gauss_opt*255).astype(np.uint8), "Gaussian Opt"),
    ((median_opt*255).astype(np.uint8), "Median Opt"),
    ((bilateral_opt*255).astype(np.uint8), "Bilateral Opt")
]):
    plt.subplot(2,2,i+1)
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")
plt.show()

