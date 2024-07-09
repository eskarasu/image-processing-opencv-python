import cv2
from matplotlib import pyplot as plt

# EXTRACTING PIXEL VALUES AND COLOR SPACE OF THE IMAGE

# Load the image
img = cv2.imread('image.png')

# Print the pixel values of the image
print('Pixel Values:\n', img)

# Print the color space of the image (BGR)
print('Color Space (BGR):\n', img[0, 0])

# Display the image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# ROTATING THE IMAGE

# Create the rotation matrix
(height, width) = img.shape[:2]
center = (width / 2, height / 2)
matrix = cv2.getRotationMatrix2D(center, 90, 1.0)  # Rotate by 90 degrees

# Rotate the image
rotated = cv2.warpAffine(img, matrix, (width, height))

# Display the rotated image
cv2.imshow('Rotated Image', rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ROTATING WITH ACCURATE DIMENSIONS

# Create the rotation matrix
(height, width) = img.shape[:2]
center = (width/2, height/0.95) # New center
matrix = cv2.getRotationMatrix2D(center, 90, 1.0)  # Rotate by 90 degrees

# Rotate the image and specify new dimensions
rotated = cv2.warpAffine(img, matrix, (height, width))  # New dimensions: height x width

# Display the rotated image
cv2.imshow('Rotated Image', rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

# RESIZING THE IMAGE

# Resize the image
resized = cv2.resize(img, (200, 600))  # New size: 200x600

# Display the resized image
cv2.imshow('Resized Image', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# CROPPING THE IMAGE

# Crop the image
cropped = img[50:200, 100:300]  # From row 50 to 200 and column 100 to 300

# Display the cropped image
cv2.imshow('Cropped Image', cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()

# BLURRING THE IMAGE

# Blur the image
blurred = cv2.blur(img, (5, 5))  # Use a 5x5 kernel

# Display the blurred image
cv2.imshow('Blurred Image', blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()

# CONVERTING THE IMAGE TO GRAYSCALE

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect edges
edges = cv2.Canny(gray, 30, 100)  # minVal = 30, maxVal = 100

# Display the edges
cv2.imshow('Edge Image', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image and convert it to grayscale
img = cv2.imread('image.png', 0)

# Compute the Fourier transform
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Display the spectrum
magnitude_spectrum = 20*np.log(np.abs(fshift))
plt.subplot(121), plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

# The image represents a "Magnitude Spectrum". The magnitude spectrum is a type of graph that shows the magnitudes of different frequencies in a signal or image. The spectrum appears as a grayscale image with variable intensities, where brighter areas represent higher magnitudes (or powers) of the corresponding frequencies. The bright vertical line in the center indicates a significant amount of energy at a particular frequency. This type of spectrum is important in fields such as signal processing and image analysis, as it helps to understand the frequency content of signals or images. This can be important for tasks such as filtering, compression, or enhancement.

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('image.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform edge detection
edges = cv2.Canny(gray, 100, 200, apertureSize=3)

# Detect lines with Hough transform
lines = cv2.HoughLines(edges, 1, np.pi / 180, 250)

# Draw the detected lines on the image
if lines is not None:
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Display the image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Detected Lines')
plt.show()

# Load the image
image = cv2.imread('image.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)

# Detect circles with Hough transform
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                            param1=60, param2=40, minRadius=10, maxRadius=50)

# Draw the detected circles on the image
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Draw the center of the circle
        cv2.circle(image, (i[0], i[1]), 1, (0, 100, 100), 3)
        # Draw the circle itself
        cv2.circle(image, (i[0], i[1]), i[2], (255, 0, 255), 3)

# Display the image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Detected Circles')
plt.show()

import cv2
import numpy as np

# Load the image and convert it to grayscale
img = cv2.imread('image.png', 0)

# Dilation
kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(img, kernel, iterations = 1)

# Display the dilated image
cv2.imshow('Dilated Image', dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Erosion
erosion = cv2.erode(img, kernel, iterations = 1)

# Display the eroded image
cv2.imshow('Eroded Image', erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2

# Load the image and convert it to grayscale
img = cv2.imread('image.png', 0)

# Thresholding
ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

cv2.imshow('Thresholded Image', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
