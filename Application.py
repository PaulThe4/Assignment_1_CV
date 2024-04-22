from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', dimensions=None)

@app.route('/measure', methods=['POST'])
def measure_dimensions():

    # Check if 'image' and 'known_dimensions' are in the request
    if 'image' not in request.files or 'known_dimensions' not in request.form:
        return "Missing image or known_dimensions in the request", 400
    
    # Receive image and selected points from frontend
    image = request.files['image']
    known_dimensions = int(request.form['known_dimensions'])  # Assuming known dimensions are sent as a string

    # Load the image
    image_np = np.fromfile(image, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Define the selected corresponding points (pixel coordinates)
    image_points = np.array([(772, 1861), (2292, 1936), (1543, 1253), (1541, 1254)])  # end to end diameter and some other points

    # Define the corresponding real-world coordinates for each pixel coordinate
    real_world_points = np.array([(0, 0), (known_dimensions, 0), (0, known_dimensions), (known_dimensions, known_dimensions)])

    # Compute the homography matrix
    homography, _ = cv2.findHomography(image_points, real_world_points)

    # Define the image coordinates of the object's boundaries
    image_boundary_points = np.array([(0, 0), (image.shape[1], 0), (image.shape[1], image.shape[0]), (0, image.shape[0])], dtype=np.float32)

    # Project the image boundary points onto the real-world plane
    real_world_boundary_points = cv2.perspectiveTransform(image_boundary_points.reshape(-1, 1, 2), homography)

    # Reshape the real_world_boundary_points array
    real_world_boundary_points = real_world_boundary_points.reshape(4, 2)

    # Convert real_world_boundary_points to a nested list of float values using tolist()
    real_world_boundary_points_list = real_world_boundary_points.tolist()

    # Calculate the distance between corresponding real-world boundary points
    real_world_distance = np.linalg.norm(real_world_boundary_points[0] - real_world_boundary_points[1])

    print("Estimated real-world dimensions of the object:", real_world_distance, "cm")

    # Render a new HTML page with the result
    return render_template('index.html', dimensions=real_world_distance)

if __name__ == '__main__':
    app.run(debug=True)