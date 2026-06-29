import cv2
import numpy as np
import os

output_path = os.path.join("static", "demo.mp4")

# Create a 3-second 112x112 video at 25 fps
fps = 25
duration = 3
frames = fps * duration
width, height = 112, 112

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

for i in range(frames):
    # Create some moving patterns to simulate a demo video
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Moving circle
    x = int(width / 2 + np.sin(i * 0.2) * 30)
    y = int(height / 2 + np.cos(i * 0.2) * 30)
    cv2.circle(frame, (x, y), 15, (100, 200, 100), -1)
    
    # Add text
    cv2.putText(frame, f"F:{i}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    out.write(frame)

out.release()
print(f"Generated synthetic demo video at {output_path}")
