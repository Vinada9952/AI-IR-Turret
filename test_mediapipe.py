import mediapipe
import cv2
import numpy as np

mp = mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize video capture
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print( "setup done" )

def captureImage():
    """Capture a frame from the video feed"""
    global cap
    ret, frame = cap.read()
    if not ret:
        return False, None
    return ret, frame

def drawSkeleton(frame):
    """Process frame and draw skeleton landmarks"""
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Mirror the frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        
        # Process the frame
        results = pose.process(rgb)
        
        # Convert back to BGR
        rgb.flags.writeable = True
        output = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        avg_torso = None
        
        if results.pose_landmarks:
            # Draw the pose landmarks
            mp_drawing.draw_landmarks(
                output,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
            )
            
            # Extract torso points
            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            
            # Calculate average torso position
            avg_torso = (
                (left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) / 4,
                (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4
            )
            
            # Draw torso center point
            h, w = output.shape[:2]
            center_point = (int(avg_torso[0] * w), int(avg_torso[1] * h))
            cv2.circle(output, center_point, 5, (255, 0, 0), -1)
            
        return output, avg_torso

try:
    while True:
        ret, frame = captureImage()
        if not ret:
            print("Failed to grab frame")
            break

        img, pos = drawSkeleton(frame)
        if pos:
            print( f"torso : {pos}" )

        cv2.imshow("Webcam - Skeleton", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
