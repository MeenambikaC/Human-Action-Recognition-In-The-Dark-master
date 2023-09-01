import cv2

# Specify the path to your video file
video_path = 'static/uploads/Walk_4_57_output.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get the width and height of the video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Print the width and height
print(f"Width: {frame_width}")
print(f"Height: {frame_height}")

# Release the video object
cap.release()
