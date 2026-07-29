import cv2 as cv
import numpy as np

#Test video loaded
cap = cv.VideoCapture("test_videos/solidWhiteRight.mp4")
output_path = "ExtractedContent/"
masking_coordinates = [np.array([[0,540], [960,540], [960, 380], [600,270], [300,270], [0, 350]], np.int32)]
#read the first frame
success_flag, frame = cap.read()
print(frame.shape)
print(frame.dtype)
print(frame[0,0])

#Saved it
cv.imwrite(output_path + "first_frame.png", frame)

# Canny edge detection, grayscaling and gaussian blur

def get_edges(input_frame):
    gray = cv.cvtColor(input_frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5,5), 0)
    edges = cv.Canny(blur, 50, 150)

    cv.imwrite(output_path + "edges1_50_150.png", edges)
    cv.imshow("frame1", edges)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return edges

def masking(input_frame):
    outlined_frame = cv.polylines(img= input_frame, pts=masking_coordinates, isClosed=True, color=(0,255,0), thickness=2)
    cv.imwrite(output_path +"outl1.png", outlined_frame)

    fresh_copy = input_frame.copy()

    filled_mask = cv.fillPoly(img=fresh_copy, pts=masking_coordinates, color=(0,255,0))
    cv.imwrite(output_path + "fill1.png", filled_mask)

    cv.imshow("outl_frame", outlined_frame)
    cv.imshow("fill_frame", filled_mask)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return outlined_frame, filled_mask

def apply_mask(edges):
    mask = np.zeros(edges.shape, dtype = np.uint8)
    cv.fillPoly(mask, masking_coordinates, 255)
    masked_edges = cv.bitwise_and(edges, mask)
    cv.imwrite(output_path + "masked.png", masked_edges)
    cv.imshow("maskedFrame", masked_edges)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return masked_edges
    
def detect_lines(masked_edges):
    line_coordinates = cv.HoughLinesP(image= masked_edges, rho= 1, theta= np.pi/180, threshold= 30, minLineLength=20, maxLineGap=50)
    #print(line_coordinates)
    return line_coordinates

def draw_line(frame, line_coordinates):
    line_frame_cpy = frame.copy()
    for line in line_coordinates:
        x1, y1, x2, y2 = line
        print(x1, y1, x2, y2)
        cv.line(line_frame_cpy, (x1, y1), (x2, y2), color=(0,0,255), thickness=3)
    
    cv.imwrite(output_path + "lined_frame.png", line_frame_cpy)
    cv.imshow("Lined_Frame", line_frame_cpy)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return line_frame_cpy

draw_line(frame, detect_lines(apply_mask(get_edges(frame))))

#detect_lines(apply_mask(get_edges(frame)))
#get_edges(frame)
# apply_mask(get_edges(frame))