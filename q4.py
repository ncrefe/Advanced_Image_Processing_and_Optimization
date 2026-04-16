import cv2
import numpy as np


# Load the image and convert to grayscale
original_img_color = cv2.imread("bonn.jpg")  # Load 'bonn.jpg'
gray_img = cv2.cvtColor(original_img_color, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
gray_img_float = gray_img.astype(np.float64) / 255.0 # Convert to float or else cv2.filter2D will return weird stuff

print(f"Image loaded successfully. Size: {gray_img.shape}")

# Define kernels from the sheet in row-major format
kernel_1 = np.array([[0.0113,0.0838,0.0113], 
                     [0.0838, 0.6193, 0.0838], 
                     [0.0113, 0.0838, 0.0113]])

kernel_2 = np.array([[-0.8984, 0.1472, 1.1410],
                     [-1.9075, 0.1566, 2.1359],
                     [-0.8659, 0.0573, 1.0337]])


NUMERIC_EPS = 1e-10
# (a) Decompose the Kernels via cv2 SVD
def check_decomp(kernel):
    w, u, vt = cv2.SVDecomp(kernel)
    w = w.flatten()

    if w[0] != 0.0 and np.all(w[1:] < NUMERIC_EPS): 
        print("The kernel is separable.")
        return True, w, u, vt
    else: 
        # b) 
        print("The kernel is NOT separable.")
        return False, w, u, vt
    

sep_1, w_1, u_1, vt_1 = check_decomp(kernel_1)
sep_2, w_2, u_2, vt_2= check_decomp(kernel_2)


# b) 
def approximate_SVD(w,u,vt): 
    sigma = w[0]
    u_vector = u[:,0] 
    v_vector = vt[0,:] # is this already transposed ? 

    return sigma * np.outer(u_vector, v_vector)

approx_1 = None
if not sep_1: 
    approx_1 = approximate_SVD(w_1,u_1, vt_1)

approx_2 = None
if not sep_2: 
    approx_2 = approximate_SVD(w_2, u_2, vt_2)


kerne1_result = cv2.filter2D(gray_img_float, -1, kernel_1)
if approx_1 is not None: 
    approx1_result = cv2.filter2D(gray_img_float, -1, approx_1)
    # c) Pixel-wise difference
    diff_1 = np.abs(kerne1_result-approx1_result)
    print(f"MAX Difference for Kernel 1 is {np.max(diff_1)}")



kernel2_result = cv2.filter2D(gray_img_float, -1, kernel_2)
if approx_2 is not None: 
    approx2_result = cv2.filter2D(gray_img_float, -1, approx_2)
    # c) Pixel-wise difference
    diff_2 = np.abs(kernel2_result-approx2_result)
    print(f"MAX Difference for Kernel 2 is {np.max(diff_2)}")
