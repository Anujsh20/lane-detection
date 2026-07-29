import cv2 as cv

cap = cv.VideoCapture("test_videos/solidWhiteRight.mp4")

success_flag, frame = cap.read()
fd = frame.shape
print(fd)
print(frame.dtype)
print(frame[0,0])

cv.imwrite("first_frame.png", frame)

def ced(frame):
    #img = cv.imread("first_frame.png")
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5,5), 0)
    edges1 = cv.Canny(blur, 50, 150)
    edges2 = cv.Canny(blur, 10, 250)

    cv.imwrite("edges1_50_150.png", edges1)
    cv.imwrite("edges2_10_250.png", edges2)

    cv.imshow("frame1", edges1)
    cv.imshow("frame2", edges2)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return edges1, edges2

ced(frame)