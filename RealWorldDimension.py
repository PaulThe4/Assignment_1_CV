import cv2
import numpy as np

# Load the image
image = cv2.imread("/Users/sonipriyapaul/Downloads/Assignment_1_CV/Ball_70mm.jpg")

# Display the image
cv2.imshow("Ball Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Define the selected corresponding points (pixel coordinates)
image_points = np.array([(772, 1861), (2292, 1936), (1543, 1253), (1541, 1254)])  # end to end diamter and some other points

# Define the corresponding real-world coordinates
# Real World diameter of ball in cm
real_world_diameter = 7.0  

# Define the corresponding real-world coordinates for each pixel coordinate
real_world_points = np.array([(0, 0), (real_world_diameter, 0), (0, real_world_diameter), (real_world_diameter, real_world_diameter)])

# Compute the homography matrix
homography, _ = cv2.findHomography(image_points, real_world_points)

# Define the image coordinates of the object's boundaries
# This could be the bounding box coordinates or the image dimensions
image_boundary_points = np.array([(0, 0), (image.shape[1], 0), (image.shape[1], image.shape[0]), (0, image.shape[0])], dtype=np.float32)

# Project the image boundary points onto the real-world plane
real_world_boundary_points = cv2.perspectiveTransform(image_boundary_points.reshape(-1, 1, 2), homography)

print("Shape of real_world_boundary_points:", real_world_boundary_points.shape)

# Reshape the real_world_boundary_points array to (4, 2)
real_world_boundary_points = real_world_boundary_points.reshape(4, 2)

# Calculate the distance between corresponding real-world boundary points
# This will give you the estimated real-world dimensions of the object
real_world_distance = np.linalg.norm(real_world_boundary_points[0] - real_world_boundary_points[1])

print("Estimated real-world dimensions of the ball:", real_world_distance)