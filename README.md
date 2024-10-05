# EX-05: Implementation-of-filter
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step-1:Load Image:
Read the original image in grayscale using OpenCV.
### Step-2:Add Gaussian Noise:
Generate Gaussian noise using a mean of 0 and a specified standard deviation (e.g., 25).
Add the generated noise to the original image and clip the values to ensure they remain in the valid range (0-255).
### Step-3:Define Filter Kernel:
Create a 3x3 weighted average kernel (e.g., [[1, 2, 1], [2, 4, 2], [1, 2, 1]]), normalizing it by dividing by the sum of the weights (16).
### Step-4:Pad the Image:
Create a padded version of the noisy image to handle border effects during convolution.
### Step-5:Apply Convolution:
Loop through each pixel in the padded image, extract the corresponding region of interest (ROI), and apply the filter by performing an element-wise multiplication and summing the result. Store the filtered values in a new output image.
### Step-6:Display Results:
Use Matplotlib to display the original image, the noisy image, and the filtered image side by side for comparison

## Program:
### Developed By   : Prajin S
### Register Number: 212223230151


### 1. Smoothing Filters

i) Using Averaging Filter
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('ex-0412.jpg', cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(12, 4))
```
![image](https://github.com/user-attachments/assets/5e8199ce-1311-4871-9037-03712a810aa3)
```
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')
```
![image](https://github.com/user-attachments/assets/dbca059e-d9f9-49ac-9d1a-dde18592709c)
```
gaussian_noise = np.random.normal(0,25, image.shape)
noisy_image = image + gaussian_noise
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image (Gaussian Noise)')
plt.axis('off')
```
![image](https://github.com/user-attachments/assets/4f4eefee-7fa4-4b97-981b-f9d51f5fc211)
```
filtered_image = np.zeros_like(noisy_image)
height, width = noisy_image.shape
for i in range(1, height - 1):
    for j in range(1, width - 1):
        neighborhood = noisy_image[i - 1:i + 2, j - 1:j + 2]
        filtered_value = np.mean(neighborhood)
        filtered_image[i, j] = filtered_value
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image (Manual Box Filter 3x3)')
plt.axis('off')
```
![image](https://github.com/user-attachments/assets/0ed15aac-53ef-47f4-afb8-ac29b02dde2c)

ii) Using Weighted Averaging Filter
```
image = cvplt.imshow(image, cmap='gray')
plt.imshow(image,cmap="gray")
plt.title('Original Image')
plt.axis('off')
```
![image](https://github.com/user-attachments/assets/8747e2e0-cd79-43cc-b10c-12d62927ac90)
```
gaussian_noise = np.random.normal(0,25, image.shape)
noisy_image = image + gaussian_noise
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image (Gaussian Noise)')
plt.axis('off')
```
![image](https://github.com/user-attachments/assets/7936918a-38db-44c3-96e6-fe599f1deb97)
```
kernel = np.array([[1, 2, 1],
                   [2, 4, 2],
                   [1, 2, 1]]) / 16.0  # Normalize the kernel

image_height, image_width = noisy_image.shape
kernel_size = kernel.shape[0]  
pad = kernel_size // 2

padded_image = np.pad(noisy_image, pad, mode='constant', constant_values=0)

filtered_image = np.zeros_like(noisy_image)

for i in range(pad, image_height + pad):
    for j in range(pad, image_width + pad):
        roi = padded_image[i - pad:i + pad + 1, j - pad:j + pad + 1]
        filtered_value = np.sum(roi * kernel)
        filtered_image[i - pad, j - pad] = np.clip(filtered_value, 0, 255)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image (Manual Weighted Avg)')
plt.axis('off')
```
![image](https://github.com/user-attachments/assets/b6a13087-0db0-41d5-b55c-e4e575fbdf55)

iii) Using Minimum Filter
```
noisy_image = np.copy(image)
salt_prob = 0.05  
pepper_prob = 0.05  
noisy_image = np.copy(image)
num_salt = np.ceil(salt_prob * image.size)
coords_salt = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
noisy_image[tuple(coords_salt)] = 255
num_pepper = np.ceil(pepper_prob * image.size)
coords_pepper = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
noisy_image[tuple(coords_pepper)] = 0
```
![image](https://github.com/user-attachments/assets/c3487475-20e5-454a-9cb2-fdddcddf3bc0)

```
min_filtered_image = np.zeros_like(noisy_image)
max_filtered_image = np.zeros_like(noisy_image)
med_filtered_image = np.zeros_like(noisy_image)
height, width = noisy_image.shape
for i in range(1, height - 1):
    for j in range(1, width - 1):
        neighborhood = noisy_image[i - 1:i + 2, j - 1:j + 2]
        min_filtered_image[i, j] = np.min(neighborhood)
min_filtered_image = np.zeros_like(noisy_image)
plt.imshow(min_filtered_image, cmap='gray')
plt.title('Filtered Image (Min Filter)')
plt.axis('off')
```
![image](https://github.com/user-attachments/assets/fb99c274-effe-4fef-8162-c4772f687c35)


iv) Using Maximum Filter
```

for i in range(1, height - 1):
    for j in range(1, width - 1):
        neighborhood = noisy_image[i - 1:i + 2, j - 1:j + 2]
        min_filtered_image[i, j] = np.min(neighborhood)
max_filtered_image = np.zeros_like(noisy_image)
plt.imshow(max_filtered_image, cmap='gray')
plt.title('Filtered Image (Max Filter)')
plt.axis('off')
```
![image](https://github.com/user-attachments/assets/cf4d188d-b07c-4531-a397-ec846ceecda8)


v) Using Median Filter
```
for i in range(1, height - 1):
    for j in range(1, width - 1):
        neighborhood = noisy_image[i - 1:i + 2, j - 1:j + 2]
        med_filtered_image[i, j] = np.median(neighborhood)
plt.imshow(med_filtered_image, cmap='gray')
plt.title('Filtered Image (Med Filter)')
plt.axis('off')
```
![image](https://github.com/user-attachments/assets/d3967330-5b5a-45cb-b8ab-d9aa5dbaa78b)

### 2. Sharpening Filters
i) Using Laplacian Linear Kernal
```
image = cv2.imread('ex-0412.jpg', cv2.IMREAD_GRAYSCALE)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')
```
![image](https://github.com/user-attachments/assets/34fbc329-8c48-4e65-97cf-74ec3cec593d)
```
blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
laplacian_kernel = np.array([[0, -1, 0],
                             [-1, 4, -1],
                             [0, -1, 0]])
image_height, image_width = blurred_image.shape
kernel_height, kernel_width = laplacian_kernel.shape
pad_height = kernel_height // 2
pad_width = kernel_width // 2
padded_image = np.pad(blurred_image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
```
![image](https://github.com/user-attachments/assets/b3e9b70b-d6a8-4cb9-a568-4a4a1604f3cd)

ii) Using Laplacian Operator
```
laplacian_image = np.zeros_like(blurred_image)
for i in range(image_height):
    for j in range(image_width):
        region = padded_image[i:i + kernel_height, j:j + kernel_width]
        laplacian_value = np.sum(region * laplacian_kernel)
        laplacian_image[i, j] = laplacian_value
laplacian_image = np.clip(laplacian_image, 0, 255).astype(np.uint8)
plt.imshow(laplacian_image, cmap='gray')
plt.title('Laplacian Filtered Image')
plt.axis('off')
```
![image](https://github.com/user-attachments/assets/c97c0ccb-f8dd-4fc3-86fb-5d16fa09467c)
```
sharpened_image = cv2.add(image, laplacian_image)
plt.imshow(sharpened_image, cmap='gray')
plt.title('Sharpened Image')
plt.axis('off')
```
![image](https://github.com/user-attachments/assets/9d63575a-02ba-423a-9dee-4e032e4e7eb2)

## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
