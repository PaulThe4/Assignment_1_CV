import cv2

# Load the image
image = cv2.imread("/Users/sonipriyapaul/Downloads/Assignment_1_CV/Ball_70mm.jpg")

# Function to handle mouse click events
def mouse_click(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Pixel coordinates (x, y):", x, y)

# Display the image and set the mouse callback function
cv2.imshow("Image", image)
cv2.setMouseCallback("Image", mouse_click)

# Wait for the user to click on points
cv2.waitKey(0)
cv2.destroyAllWindows()