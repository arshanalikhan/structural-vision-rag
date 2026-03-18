import cv2
import numpy as np
import requests
import torch
from ultralytics import YOLO
from scipy.spatial import distance as dist

# 1. HARDWARE SAFETY LIMITS (Crucial for 16GB RAM laptops)
# Force PyTorch to only use 2 CPU threads so it doesn't freeze Windows
torch.set_num_threads(2)

# 2. LOAD CUSTOM BRAIN
model = YOLO('best.pt') 

# 3. CAPTURE THE FRAME
frame = cv2.imread('concrete_crack.jpg')

# 4. RUN THE AI (Optimized for low RAM)
# imgsz=320 drastically reduces RAM usage, and device="cpu" forces safe execution
results = model.predict(frame, imgsz=320, device="cpu")

# 5. THE MATHEMATICAL BRIDGE
if results[0].masks is not None:
    # Extract the boundary pixels (contour) of the crack
    mask = results[0].masks.xy[0].astype(np.int32)
    
    # Calculating the bounding edges
    left_edge = tuple(mask[mask[:, 0].argmin()])
    right_edge = tuple(mask[mask[:, 0].argmax()])
    
    # Calculating Euclidean distance
    pixel_distance = dist.euclidean(left_edge, right_edge)
    
    # Applying pixel-to-metric ratio (Assuming 1 pixel = 0.5 mm)
    pixels_per_metric = 2.0 
    crack_width_mm = pixel_distance / pixels_per_metric
    
    print(f"\n--- HARDWARE MEASUREMENT ---")
    print(f"Crack detected at {crack_width_mm:.2f} mm")

    # 6. THE COMPLIANCE AUDIT
    generated_query = "What is the maximum permissible crack width for concrete?"
    print(f"Querying IS 456 Database: '{generated_query}'")
    
    # Sending the query to existing Flask backend
    flask_url = "http://127.0.0.1:5000/ask"
    
    try:
        response = requests.post(flask_url, json={"query": generated_query})
        
        if response.status_code == 200:
            is456_rule = response.json()
            print("\n--- AUDIT RESULT ---")
            print(f"Rule: Section {is456_rule['section']} - {is456_rule['title']}")
            
            # Logic Check
            if crack_width_mm > 0.3:
                print(f"STATUS: ❌ FAIL - {crack_width_mm:.2f} mm exceeds permissible limits.")
            else:
                print(f"STATUS: ✅ PASS - {crack_width_mm:.2f} mm is within safe limits.")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the NLP brain. Is app.py running?")
        
else:
    print("No cracks detected in the frame.")