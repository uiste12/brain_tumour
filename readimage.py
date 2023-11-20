import cv2

# Path to the image file
image_path = 'C:/Users/User/Desktop/Br35H/datasets/no/no0.jpg'

# Read the image using cv2
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is not None:
    # Display the image (optional)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Image not found or could not be loaded.")
