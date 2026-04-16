import cv2
import numpy as np
import time

# ==============================================================================
# 0. Setup: Loading Image and Converting to Grayscale
# ==============================================================================
print("--- 0. Setup: Loading Image and Converting to Grayscale ---")

'''
TODO: Load the image 'bonn.jpg' and convert it to grayscale
'''

# Load image and convert to grayscale
original_img_color = cv2.imread("bonn.jpg")  # Load 'bonn.jpg'
gray_img = cv2.cvtColor(original_img_color, cv2.COLOR_BGR2GRAY)            # Convert to grayscale

print(f"Image loaded successfully. Size: {gray_img.shape}")

# ==============================================================================
# 1. Calculate Integral Image (Part a)
# ==============================================================================
print("\n--- a) Calculating Integral Image ---")


def calculate_integral_image(img):
    """
    Calculate the integral image (summed area table).
    Each pixel contains the sum of all pixels above and to the left.
    
    Args:
        img: Input grayscale image
    
    Returns:
        Integral image with dimensions (height+1, width+1)
    
    TODO:
    1. Create an integral image array     
    2. Iterate through all pixels and compute integral values
    
        """
    img = np.array(img, dtype=np.float64)
    h, w = img.shape
    integral_image = np.zeros((h+1, w+1), dtype=np.float64)

    # Start from range 1 to use recursion formular and be well defined 
    for i in range(1, h+1):
        for j in range(1, w+1): 
            # Recursive definition of the integral image
            integral_image[i,j] = (
                integral_image[i-1][j]+integral_image[i,j-1] - integral_image[i-1][j-1] + img[i-1][j-1] # img has to be i-1,j-1 else we would ignore the extra dim added in the integral
            ) 

    return integral_image


# Calculate integral image
integral_img = calculate_integral_image(gray_img)  # Call calculate_integral_image()

print("Integral image calculated successfully.")
print(f"Integral image size: {integral_img.shape}")

# ==============================================================================
# 2. Compute Mean Using Integral Image (Part b)
# ==============================================================================
print("\n--- b) Computing Mean Using Integral Image ---")


def mean_using_integral(integral, top_left, bottom_right):
    """
    Calculate mean gray value using integral image.
    Time Complexity: O(1)

    Args:
        integral: The integral image
        top_left: (row, col) - top left corner of the region
        bottom_right: (row, col) - bottom right corner of the region
    
    Returns:
        Mean gray value of the region
    
    TODO:
    1. Extract coordinates from top_left and bottom_right
    2. Adjust indices for integral image (remember it's 1-indexed)
    3. Return Sum / number_of_pixels
    """
    row1, col1 = top_left
    # bottom-right corner
    row2, col2 = bottom_right

    sum_region = integral[row2+1, col2+1] - integral[row1, col2+1] \
                - integral[row2+1, col1] + integral[row1, col1]

    N = (row2 - row1 + 1) * (col2 - col1 + 1)
    return sum_region / N


# Define region
top_left = (10, 10)
bottom_right = (60, 80)

# Calculate mean using integral image
mean_integral = mean_using_integral(integral_img, top_left, bottom_right)  # Call mean_using_integral()

print(f"Region: Top-left {top_left}, Bottom-right {bottom_right}")
print(f"Region size: {bottom_right[0] - top_left[0] + 1} x {bottom_right[1] - top_left[1] + 1} pixels")
print(f"Mean gray value (Integral Image Method): {mean_integral:.2f}")

# ==============================================================================
# 3. Compute Mean by Direct Summation (Part c)
# ==============================================================================
print("\n--- c) Computing Mean by Direct Summation ---")


def mean_by_direct_sum(img, top_left, bottom_right):
    """
    Calculate mean gray value by summing all pixels in region.
    Time Complexity: O(w * h) where w and h are region dimensions

    Args:
        img: The grayscale image
        top_left: (row, col) - top left corner of the region
        bottom_right: (row, col) - bottom right corner of the region
    
    Returns:
        Mean gray value of the region
    
    TODO:
    1. Extract the region from the image using array slicing
    2. Calculate and return the mean of all pixels in the region
    
      """
    img = np.array(img, dtype=np.float64)
    area = img[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
    return np.mean(area)


# Calculate mean using direct summation
mean_direct = mean_by_direct_sum(gray_img, top_left, bottom_right)  # Call mean_by_direct_sum()

print(f"Mean gray value (Direct Summation Method): {mean_direct:.2f}")

# ==============================================================================
# 4. Analyze Computational Complexity (Part d)
# ==============================================================================
print("\n--- d) Computational Complexity Analysis ---")

'''
TODO:
1. Benchmark both methods by running them multiple times (e.g., 100 iterations)
2. Measure execution time for both methods using time.perf_counter()
3. Compare the execution times
4. Verify that both methods produce the same result
5. Print the results:
   - Method name
   - Average execution time
   - Performance improvement factor


'''

# Benchmark parameters
iterations = 100

print(f"\nBenchmarking with {iterations} iterations...\n")

# TODO: Implement benchmarking code here
start_integral = time.perf_counter()
for _ in range(iterations): 
    integral_result = mean_using_integral(integral_img, top_left, bottom_right)
stop_integral = time.perf_counter()
time_integral = stop_integral-start_integral
time_integral_per_instance = time_integral / 100

start_direct = time.perf_counter()
for _ in range(iterations): 
    direct_result = mean_by_direct_sum(gray_img, top_left, bottom_right)
stop_direct = time.perf_counter()
time_direct = stop_direct-start_direct
time_direct_per_instance = time_direct / 100

assert np.isclose(direct_result, integral_result)

# TODO: Display results 
# TODO: Print theoretical complexity explanation

print("\nMethod: Direct Sum")
print(f"Average Execution Time: {time_direct_per_instance:.6f} s")
print(f"Total Execution Time: {time_direct:.6f} s")

print("\nMethod: Integral Sum")
print(f"Average Execution Time: {time_integral_per_instance:.6f} s")
print(f"Total Execution Time: {time_integral:.6f} s")

print(f"Speedup (Direct/Integral) {time_direct_per_instance/time_integral_per_instance:.2f}")


print("\nTheoretical Complexity (Big-O notation)")
print("Direct sum: Complexity is O(h*w) where h,w are size of the input region.")
print("Integral Image: Complexity is O(1), however we have to pre-compute the Integral image which is O(n*m).")