import cv2 as cv

cap = cv.VideoCapture("test_videos/solidWhiteRight.mp4")

success_flag, frame = cap.read()
fd = frame.shape
print(fd)
print(frame.dtype)
print(frame[0,0])

cv.imwrite("first_frame.png", frame)