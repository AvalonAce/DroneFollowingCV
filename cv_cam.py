#!/usr/bin/env python3
"""
OpenCV Camera Template with Moving Camera Calibration
Calibrates camera using stationary checkerboard from different angles
Detects red object position in 1m×1m workspace with checkerboard at center
Configured for 8x11 squares checkerboard (7x10 inner corners)
"""

import cv2
import sys
import numpy as np
import pickle
import os

os.environ['QT_QPA_PLATFORM'] = 'xcb'


class CameraCalibrator:
    """Handles camera calibration using stationary checkerboard pattern"""
    
    def __init__(self, checkerboard_size=(7, 10), square_size=0.025):
        """
        Initialize calibrator
        
        Args:
            checkerboard_size: (columns, rows) of inner corners
            square_size: size of checkerboard square in meters (default 25mm)
        """
        self.checkerboard_size = checkerboard_size
        print(f"Looking for checkerboard with {checkerboard_size[0]}x{checkerboard_size[1]} inner corners")
        print(f"This means a grid of {checkerboard_size[0]+1}x{checkerboard_size[1]+1} squares")
        self.square_size = square_size
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvec = None
        self.tvec = None
        self.calibration_file = 'camera_calibration.pkl'
        
    def calibrate_from_images(self, cap, num_images=20):
        """
        Capture images from different camera angles around stationary checkerboard
        
        Args:
            cap: OpenCV VideoCapture object
            num_images: number of calibration images to capture from different angles
        """
        # Prepare object points
        objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 
                               0:self.checkerboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        
        # Arrays to store object points and image points
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        
        print(f"\n{'='*60}")
        print("CALIBRATION MODE: Move camera to {num_images} different angles")
        print("Keep the checkerboard STATIONARY and move the camera around it")
        print("Try different angles: left, right, top, bottom, tilted views")
        print("Press SPACE to capture from each angle, 'q' to finish early")
        print(f"{'='*60}\n")
        
        captured = 0
        last_capture_corners = None
        
        while captured < num_images:
            ret, frame = cap.read()
            if not ret:
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Find checkerboard corners
            ret_corners, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
            
            display_frame = frame.copy()
            if ret_corners:
                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # Draw corners
                cv2.drawChessboardCorners(display_frame, self.checkerboard_size, 
                                         corners_refined, ret_corners)
                
                # Check if this view is significantly different from last capture
                is_different = True
                if last_capture_corners is not None:
                    # Compare corner positions to ensure camera moved
                    diff = np.linalg.norm(corners_refined - last_capture_corners)
                    if diff < 50:  # Threshold in pixels
                        is_different = False
                        cv2.putText(display_frame, "Move camera to a different angle!", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
                status_color = (0, 255, 0) if is_different else (0, 165, 255)
                cv2.putText(display_frame, f"Captured: {captured}/{num_images} - Press SPACE", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            else:
                cv2.putText(display_frame, "Checkerboard not detected - adjust camera angle", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Calibration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and ret_corners:
                objpoints.append(objp)
                imgpoints.append(corners_refined)
                last_capture_corners = corners_refined.copy()
                captured += 1
                print(f"Image {captured}/{num_images} captured from this angle!")
            elif key == ord('q'):
                break
        
        cv2.destroyWindow('Calibration')
        
        if len(objpoints) < 10:
            print("Error: Not enough calibration images captured")
            return False
        
        print("\nCalculating camera calibration...")
        ret, self.camera_matrix, self.dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
        
        if ret:
            print("Camera calibration successful!")
            print(f"Used {len(objpoints)} images from different angles")
            self.save_calibration()
            return True
        else:
            print("Camera calibration failed")
            return False
    
    def calibrate_pose_single_frame(self, frame):
        """
        Get camera pose from a single frame with checkerboard visible
        This establishes the coordinate system with checkerboard at origin
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
        
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Prepare object points - centered at (0,0)
            # This makes the checkerboard center the origin of the 1m×1m space
            objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 
                                   0:self.checkerboard_size[1]].T.reshape(-1, 2)
            objp *= self.square_size
            
            # Shift to center the checkerboard at origin
            center_x = (self.checkerboard_size[0] - 1) * self.square_size / 2
            center_y = (self.checkerboard_size[1] - 1) * self.square_size / 2
            objp[:, 0] -= center_x
            objp[:, 1] -= center_y
            
            # Find the rotation and translation vectors
            ret, rvec, tvec = cv2.solvePnP(objp, corners_refined, 
                                           self.camera_matrix, self.dist_coeffs)
            
            if ret:
                self.rvec = rvec
                self.tvec = tvec
                return True
        
        return False
    
    def save_calibration(self):
        """Save calibration parameters to file"""
        data = {
            'camera_matrix': self.camera_matrix,
            'dist_coeffs': self.dist_coeffs,
            'rvec': self.rvec,
            'tvec': self.tvec
        }
        with open(self.calibration_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Calibration saved to {self.calibration_file}")
    
    def load_calibration(self):
        """Load calibration parameters from file"""
        if os.path.exists(self.calibration_file):
            with open(self.calibration_file, 'rb') as f:
                data = pickle.load(f)
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']
            self.rvec = data.get('rvec')
            self.tvec = data.get('tvec')
            print("Calibration loaded from file")
            return True
        return False
    
    def pixel_to_world(self, pixel_x, pixel_y, z_world=0):
        """
        Convert pixel coordinates to world coordinates
        Assumes the object is on the same plane as the checkerboard (z=0)
        Checkerboard center is at (0, 0) in the 1m×1m workspace
        
        Args:
            pixel_x, pixel_y: pixel coordinates
            z_world: z coordinate in world space (default 0 for checkerboard plane)
        
        Returns:
            (x, y, z) in world coordinates (meters), or None if not calibrated
        """
        if self.camera_matrix is None or self.rvec is None or self.tvec is None:
            return None
        
        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(self.rvec)
        R_inv = R.T
        
        # Camera matrix inverse
        K_inv = np.linalg.inv(self.camera_matrix)
        
        # Pixel coordinate in homogeneous form
        uv1 = np.array([[pixel_x], [pixel_y], [1.0]])
        
        # Left side of equation: R_inv @ (K_inv @ uv1)
        left_side = R_inv @ (K_inv @ uv1)
        
        # Right side: R_inv @ tvec
        right_side = R_inv @ self.tvec
        
        # Solve for scale s using the plane equation (z_world = z_plane)
        # We know that world_z = s * left_side[2] - right_side[2] = z_world
        # Therefore: s = (z_world + right_side[2]) / left_side[2]
        s = (z_world + right_side[2, 0]) / left_side[2, 0]
        
        # Calculate world coordinates
        world_point = s * left_side - right_side
        
        return world_point.flatten()
    
    def is_in_workspace(self, x, y, workspace_size=1.0):
        """
        Check if coordinates are within the 1m×1m workspace
        
        Args:
            x, y: world coordinates in meters
            workspace_size: size of workspace in meters (default 1.0m)
        
        Returns:
            True if within workspace bounds
        """
        half_size = workspace_size / 2
        return abs(x) <= half_size and abs(y) <= half_size


def main():
    # Find and open the camera
    for i in range(1, 10):
        cap = cv2.VideoCapture()
        cap.open(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera found at index {i}, resolution: {frame.shape}")
                break
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)
    
    print("Camera opened successfully!")
    
    # Initialize calibrator for 8x11 squares pattern
    # 8x11 squares = 7x10 inner corners
    # square_size is in meters (e.g., 0.025 = 25mm squares)
    calibrator = CameraCalibrator(checkerboard_size=(7, 10), square_size=0.025)
    
    # Diagnostic mode - try different orientations
    print("\n" + "="*60)
    print("DIAGNOSTIC MODE")
    print("If checkerboard is not detected, try rotating it:")
    print("Press '1' for 7x10 inner corners (8x11 squares)")
    print("Press '2' for 10x7 inner corners (11x8 squares) - rotated 90°")
    print("Press ENTER to continue with current size")
    print("="*60)
    
    test_mode = True
    while test_mode:
        ret, frame = cap.read()
        if not ret:
            continue
        
        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Try to detect with current size
        ret_corners, corners = cv2.findChessboardCorners(gray, calibrator.checkerboard_size, 
                                                         cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                         cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if ret_corners:
            cv2.drawChessboardCorners(display_frame, calibrator.checkerboard_size, corners, ret_corners)
            status_text = f"DETECTED! {calibrator.checkerboard_size[0]}x{calibrator.checkerboard_size[1]} corners ({calibrator.checkerboard_size[0]+1}x{calibrator.checkerboard_size[1]+1} squares)"
            cv2.putText(display_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            status_text = f"NOT DETECTED - Trying {calibrator.checkerboard_size[0]}x{calibrator.checkerboard_size[1]} corners"
            cv2.putText(display_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_frame, "Try rotating pattern (press 1-2) or ENTER to continue", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow('Diagnostic Mode', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            calibrator.checkerboard_size = (7, 10)
            print(f"Trying {calibrator.checkerboard_size} inner corners (8x11 squares)")
        elif key == ord('2'):
            calibrator.checkerboard_size = (10, 7)
            print(f"Trying {calibrator.checkerboard_size} inner corners (11x8 squares)")
        elif key == 13 or key == 10:  # ENTER key
            test_mode = False
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
    
    cv2.destroyWindow('Diagnostic Mode')
    
    # Try to load existing calibration
    if not calibrator.load_calibration():
        print("\nNo existing calibration found. Starting calibration process...")
        print("\nIMPORTANT: Keep the checkerboard STATIONARY!")
        print("You will move the CAMERA to different angles around it.")
        if not calibrator.calibrate_from_images(cap, num_images=20):
            print("Calibration failed. Exiting.")
            cap.release()
            sys.exit(1)
    
    # Establish coordinate system by detecting checkerboard
    print("\n" + "="*60)
    print("POSE CALIBRATION: Position camera to see the checkerboard")
    print("The checkerboard center will be the origin (0, 0) of the 1m×1m workspace")
    print("Press SPACE when checkerboard is detected to set coordinate system")
    print("="*60)
    
    pose_calibrated = False
    while not pose_calibrated:
        ret, frame = cap.read()
        if not ret:
            continue
        
        display_frame = frame.copy()
        
        # Try to detect checkerboard
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_corners, corners = cv2.findChessboardCorners(gray, calibrator.checkerboard_size, 
                                                         cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                         cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                         cv2.CALIB_CB_FAST_CHECK)
        
        if ret_corners:
            cv2.drawChessboardCorners(display_frame, calibrator.checkerboard_size, corners, ret_corners)
            cv2.putText(display_frame, "Checkerboard detected - Press SPACE to set origin", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "Center of checkerboard = Center of 1m x 1m workspace", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(display_frame, "Searching for checkerboard...", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Pose Calibration', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and ret_corners:
            if calibrator.calibrate_pose_single_frame(frame):
                print("Coordinate system established!")
                print("Checkerboard center is at (0, 0)")
                print("Workspace spans from -0.5m to +0.5m in X and Y")
                pose_calibrated = True
                calibrator.save_calibration()
        elif key == ord('q'):
            print("Calibration cancelled")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
    
    cv2.destroyWindow('Pose Calibration')
    
    print("\n" + "="*60)
    print("DETECTION MODE - 1m × 1m Workspace")
    print("Detecting red objects within 1m × 1m space centered on checkerboard")
    print("Press 'q' to quit, 's' to save a screenshot")
    print("Press 'r' to recalibrate pose")
    print("="*60 + "\n")
    
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
            
            # Draw workspace boundary indicator
            cv2.putText(frame, "1m x 1m workspace centered on checkerboard", 
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
            
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
                        
                        # Convert to world coordinates
                        world_coords = calibrator.pixel_to_world(cx, cy, z_world=0)
                        
                        if world_coords is not None:
                            wx, wy, wz = world_coords
                            
                            # Check if within workspace
                            in_workspace = calibrator.is_in_workspace(wx, wy, workspace_size=1.0)
                            
                            # Display pixel position
                            pixel_text = f"Pixel: ({cx}, {cy})"
                            cv2.putText(frame, pixel_text, (cx - 50, cy - 60),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            
                            # Display world position (in mm for readability)
                            world_text = f"Position: ({wx*1000:.1f}, {wy*1000:.1f}) mm"
                            cv2.putText(frame, world_text, (cx - 50, cy - 40),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                            
                            # Display workspace status
                            status_color = (0, 255, 0) if in_workspace else (0, 0, 255)
                            status_text = "IN WORKSPACE" if in_workspace else "OUT OF BOUNDS"
                            cv2.putText(frame, status_text, (cx - 50, cy - 20),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
                            
                            # Print to console
                            workspace_status = "IN" if in_workspace else "OUT"
                            print(f"Red object - Pixel: ({cx}, {cy}) | "
                                  f"World: ({wx*1000:.1f}, {wy*1000:.1f}) mm from center | "
                                  f"Status: {workspace_status} of workspace")
            
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
            
            # Recalibrate pose on 'r' key press
            elif key == ord('r'):
                print("\nRecalibrating pose...")
                pose_calibrated = False
                while not pose_calibrated:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    display_frame = frame.copy()
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    ret_corners, corners = cv2.findChessboardCorners(gray, calibrator.checkerboard_size, None)
                    
                    if ret_corners:
                        cv2.drawChessboardCorners(display_frame, calibrator.checkerboard_size, corners, ret_corners)
                        cv2.putText(display_frame, "Press SPACE to set new origin", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(display_frame, "Searching for checkerboard...", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    cv2.imshow('Pose Calibration', display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(' ') and ret_corners:
                        if calibrator.calibrate_pose_single_frame(frame):
                            print("Coordinate system re-established!")
                            pose_calibrated = True
                            calibrator.save_calibration()
                    elif key == ord('q'):
                        break
                
                cv2.destroyWindow('Pose Calibration')
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Release everything when job is finished
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed")


if __name__ == "__main__":
    main()