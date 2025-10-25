#!/usr/bin/env python3
"""
OpenCV Camera Template
A simple template for opening and displaying camera feed using OpenCV
"""

import cv2
import sys
import numpy as np

def main():
    """
    Main function to open camera, create red color mask, and locate red objects
    """
    # Initialize camera capture
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)
    
    print("Camera opened successfully!")
    print("Press 'q' to quit, 's' to save a screenshot")
    
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            # Check if frame was captured successfully
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # Convert to HSV color space (better for color detection)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Define range for red color in HSV
            # Red wraps around in HSV, so we need two ranges
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            
            # Create masks for both red ranges
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            
            # Combine both masks
            mask = cv2.bitwise_or(mask1, mask2)
            
            # Apply morphological operations to reduce noise
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours of red objects
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create result image
            result = cv2.bitwise_and(frame, frame, mask=mask)
            
            # Draw contours and find positions
            if contours:
                # Find the largest contour (assuming it's the main red object)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Only process if contour is large enough (filter out noise)
                if cv2.contourArea(largest_contour) > 500:
                    # Get the center of the object 
                    
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Draw contour on original frame
                        cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
                        
                        # Draw center point
                        cv2.circle(frame, (cx, cy), 7, (255, 0, 0), -1)
                        
                        # Draw bounding box
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        # Display position text
                        position_text = f"Position: ({cx}, {cy})"
                        cv2.putText(frame, position_text, (cx - 50, cy - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        print(f"Red object detected at position: ({cx}, {cy})")
            
            # Display the frames
            cv2.imshow('Original with Detection', frame)
            cv2.imshow('Red Mask', mask)
            cv2.imshow('Filtered Result', result)

            # Wait for key press (1ms delay)
            key = cv2.waitKey(1) & 0xFF
            
            # Exit on 'q' key press
            if key == ord('q'):
                print("Quitting...")
                break
            
            # Save screenshot on 's' key press
            elif key == ord('s'):
                cv2.imwrite('screenshot.png', frame)
                cv2.imwrite('mask.png', mask)
                print("Screenshot saved!")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Release everything when job is finished
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed")


if __name__ == "__main__":
    main()