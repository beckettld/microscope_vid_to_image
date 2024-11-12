import cv2
import numpy as np

def create_partial_circular_mask(frame):
    '''
    Working
    '''

    print('Creating mask...')
    
    # Convert the frame to grayscale and apply a binary threshold
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Find all contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area and take the two largest ones
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contours = sorted_contours[:4]  # Get the four largest contours

    # Create a mask and fill in the four largest contours
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, largest_contours, -1, 255, thickness=cv2.FILLED)
    inverted_mask = cv2.bitwise_not(mask)

    print('Mask created with the two largest contours.')

    return inverted_mask

def calculate_frame_differences(video_path, frame_step=1):
    """
    Need to fix
    
    Calculates the (x_diff, y_diff) translation for every nth frame relative to the first frame.
    Returns a list of translation tuples.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return []

    # Read the first frame and create a mask
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame from video.")
        return []

    # Apply mask to the first frame
    first_mask = create_partial_circular_mask(first_frame)
    first_frame_masked = cv2.bitwise_and(first_frame, first_frame, mask=first_mask)

    # Convert first frame to grayscale and compute ORB keypoints and descriptors
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(first_frame_masked, None)
    translations = []

    frame_number = 1
    initial_diffs = []
    
    # Calculate initial translations for the first few frames without stepping
    while frame_number <= 5:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply mask to the current frame
        frame_mask = create_partial_circular_mask(frame)
        frame_masked = cv2.bitwise_and(frame, frame, mask=frame_mask)

        # Detect and match features with the first frame
        kp2, des2 = orb.detectAndCompute(frame_masked, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Calculate median translation based on matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        x_diffs = dst_pts[:, :, 0] - src_pts[:, :, 0]
        y_diffs = dst_pts[:, :, 1] - src_pts[:, :, 1]
        
        x_diff = np.median(x_diffs)
        y_diff = np.median(y_diffs)

        # Collect initial translations
        initial_diffs.append((x_diff, y_diff))
        translations.append((x_diff, y_diff))

        frame_number += 1

    # Calculate tolerance based on initial frame translations
    initial_x_diffs = [diff[0] for diff in initial_diffs]
    initial_y_diffs = [diff[1] for diff in initial_diffs]
    median_x_diff = np.median(initial_x_diffs)
    median_y_diff = np.median(initial_y_diffs)

    tolerance_x = max(10, abs(median_x_diff) * 1.5)  # Allow a tolerance 1.5x the median
    tolerance_y = max(10, abs(median_y_diff) * 1.5)

    # Process remaining frames with the specified frame step
    while True:
        for _ in range(frame_step - 1):
            cap.read()  # Read and discard frames

        ret, frame = cap.read()
        if not ret:
            break

        # Apply mask to the current frame
        frame_mask = create_partial_circular_mask(frame)
        frame_masked = cv2.bitwise_and(frame, frame, mask=frame_mask)

        # Detect and match features with the first frame
        kp2, des2 = orb.detectAndCompute(frame_masked, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Calculate median translation based on matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        x_diffs = dst_pts[:, :, 0] - src_pts[:, :, 0]
        y_diffs = dst_pts[:, :, 1] - src_pts[:, :, 1]
        
        x_diff = np.median(x_diffs)
        y_diff = np.median(y_diffs)

        # Apply tolerance limits
        x_diff = max(min(x_diff, tolerance_x), -tolerance_x)
        y_diff = max(min(y_diff, tolerance_y), -tolerance_y)

        translations.append((x_diff, y_diff))
        frame_number += 1
        print(f"Frame {frame_number} - Translation from start: (x_diff={x_diff}, y_diff={y_diff})")

    cap.release()
    return translations


def synthesize_video_to_image(video_path, translations, mask):
    """
    Working
    
    Create an empty canvas and overlay each masked frame based on calculated translations.
    """
    print(translations)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # Estimate canvas size based on video resolution and translations
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return None

    frame_height, frame_width = first_frame.shape[:2]
    canvas_height = frame_height + int(sum(abs(y) for x, y in translations))
    canvas_width = frame_width + int(sum(abs(x) for x, y in translations))
    canvas = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)

    # Overlay the frames on the canvas
    x_offset, y_offset = 0, 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame

    # Initialize the canvas with the first frame
    first_frame_rgba = cv2.cvtColor(first_frame, cv2.COLOR_BGR2BGRA)
    first_frame_rgba[:, :, 3] = mask  # Apply the mask as alpha for transparency
    canvas[y_offset:y_offset+frame_height, x_offset:x_offset+frame_width] = first_frame_rgba

    for i, (dx, dy) in enumerate(translations):
        ret, frame = cap.read()
        if not ret:
            break

        # Apply mask to the frame
        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        frame_mask = create_partial_circular_mask(frame)
        frame_rgba[:, :, 3] = frame_mask  # Set the alpha channel based on the mask

        # Update offsets for this frame
        x_offset += int(dx)
        y_offset += int(dy)

        # Overlay frame on canvas only where canvas is empty (alpha == 0)
        for y in range(frame_height):
            for x in range(frame_width):
                if frame_rgba[y, x, 3] > 0 and canvas[y + y_offset, x + x_offset, 3] == 0:
                    canvas[y + y_offset, x + x_offset] = frame_rgba[y, x]

    cap.release()
    return canvas

# Main script execution
video_path = '/video_path'
output_image_path = '/output_image.png'

# Step 1: Detect the mask and calculate frame differences
ret, first_frame = cv2.VideoCapture(video_path).read()
mask = create_partial_circular_mask(first_frame)
translations = calculate_translations(video_path)

# Step 2: Synthesize the video into a single image
final_image = synthesize_video_to_image(video_path, translations, mask)
if final_image is not None:
    cv2.imwrite(output_image_path, final_image)
    print(f"Synthesized image saved as {output_image_path}")
