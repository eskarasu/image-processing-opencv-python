import cv2  # Import the OpenCV library for computer vision tasks

# Load the original image
original_image = cv2.imread("image.png")

# Get the dimensions of the image
h, w = original_image.shape[:2]

# Resize the image
resized_image = cv2.resize(original_image, (w*3, h*3))  # Double the width and height

cv2.imwrite('new_image.png', resized_image)
