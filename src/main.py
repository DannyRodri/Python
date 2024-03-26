import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
font_color = (255, 0, 255)
line_thickness = 2
with mp_pose.Pose(
     static_image_mode=False) as pose:

     while True:
          ret, frame = cap.read()
          if ret == False:
               break
          frame = cv2.flip(frame, 1)
          height, width, _ = frame.shape
          frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          results = pose.process(frame_rgb)

          if results.pose_landmarks is not None:
               x1 = int(results.pose_landmarks.landmark[24].x * width)
               y1 = int(results.pose_landmarks.landmark[24].y * height)

               x2 = int(results.pose_landmarks.landmark[26].x * width)
               y2 = int(results.pose_landmarks.landmark[26].y * height)

               x3 = int(results.pose_landmarks.landmark[28].x * width)
               y3 = int(results.pose_landmarks.landmark[28].y * height)
               x4=int(results.pose_landmarks.landmark[12].x*width)
               y4=int(results.pose_landmarks.landmark[12].y*height)
               x5=int(results.pose_landmarks.landmark[4].x*width)
               y5=int(results.pose_landmarks.landmark[4].y*height)
               
               # Obtener las coordenadas de los landmarks
               left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
               left_shoulderx = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x
               right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
               left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y
               right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y

               p1 = np.array([x1, y1])
               p2 = np.array([x2, y2])
               p3 = np.array([x3, y3])

               l1 = np.linalg.norm(p2 - p3)
               l2 = np.linalg.norm(p1 - p3)
               l3 = np.linalg.norm(p1 - p2)
               output = frame.copy()
              # cv2.putText(frame, f'x1: {x1}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
               #cv2.putText(frame, f'x2: {x2}', (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 250), 2)
               #cv2.putText(frame, f'x3: {x3}', (x3, y3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 191, 0), 2)


               # Calcular el ángulo
               angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
               if angle >= 160 and left_shoulder < left_hip and right_shoulder < right_hip:
                    up = True
                    cv2.putText(frame, "Esta de pie", (50, 50), font, font_scale, font_color, line_thickness)
               elif angle <= 98 and left_shoulder < left_hip and right_shoulder < right_hip:
                    
                    cv2.putText(frame, "Esta sentada", (50, 50), font, font_scale, font_color, line_thickness)

               elif left_shoulder > left_hip or right_shoulder > right_hip:
                is_laying_down = True
                cv2.putText(frame, "Esta acostado", (50, 50), font, font_scale, font_color, line_thickness)
               #print("count: ", count)
               # Visualización
               aux_image = np.zeros(frame.shape, np.uint8)
               cv2.line(aux_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
               cv2.line(aux_image, (x2, y2), (x3, y3), (255, 0, 0), 5)
               cv2.line(aux_image, (x1, y1), (x3, y3), (255, 0, 0), 5)
               cv2.line(aux_image, (x1, y1), (x4, y4), (255, 0, 0), 5)
               contours = np.array([[x1, y1], [x2, y2], [x3, y3]])
               cv2.fillPoly(aux_image, pts=[contours], color=(0,0,0))

               output = cv2.addWeighted(frame, 1, aux_image, 0.8, 0)

               cv2.circle(output, (x4,y4), 6, (0, 255, 255), 4)
               cv2.circle(output, (x1, y1), 6, (0, 255, 255), 4)
               cv2.circle(output, (x2, y2), 6, (0, 255, 255), 4)
               cv2.circle(output, (x3, y3), 6, (0, 255, 255), 4)
               cv2.putText(output, str(int(angle)), (x2 + 30, y2), 1, 1.5, (128, 0, 250), 2)
               cv2.imshow("output", output)
          cv2.imshow("Frame", frame)
          if cv2.waitKey(1) & 0xFF == 27:
               break
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()