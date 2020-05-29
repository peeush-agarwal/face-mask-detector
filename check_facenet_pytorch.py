from facenet_pytorch import MTCNN
import torch
import numpy as np
import cv2

mtcnn = MTCNN(keep_all=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while cap.isOpened():
    ret, frame = cap.read()
    if frame is None:
        break
    
    boxes, _ = mtcnn.detect(frame)

    # Draw faces
    if boxes is not None:
        for box in boxes:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()