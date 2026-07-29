import cv2 as cv
import numpy as np

#Test video loaded
cap = cv.VideoCapture("test_videos/solidWhiteRight.mp4")
output_path = "ExtractedContent/"
masking_coordinates = [np.array([[0,540], [960,540], [960, 380], [600,270], [300,270], [0, 350]], np.int32)]
#read the first frame
success_flag, frame = cap.read()
# print(frame.shape)
# print(frame.dtype)
# print(frame[0,0])


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
        cv.line(line_frame_cpy, (x1, y1), (x2, y2), color=(0,0,255), thickness=3)
    
    cv.imwrite(output_path + "lined_frame.png", line_frame_cpy)
    cv.imshow("Lined_Frame", line_frame_cpy)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return line_frame_cpy

def extrapolation(line_coordinates):
    left_lane = []
    right_lane = []
    left_slope = []
    right_slope = []
    left_intercept = []
    right_intercept = []

    for line in line_coordinates:
        x1, y1, x2, y2 = line
        #print(x1, y1, x2, y2)
        slope = (y2 - y1)/(x2 - x1)
        intercept = y1 - (slope * x1)

        if slope <0 and abs(slope) > 0.5:
            left_lane.append(line)
            left_slope.append(slope)
            left_intercept.append(intercept)
        elif slope > 0 and abs(slope) > 0.5:
            right_lane.append(line)
            right_slope.append(slope)
            right_intercept.append(intercept)

    avg_left_slope = sum(left_slope)/len(left_slope)
    avg_right_slope = sum(right_slope)/len(right_slope)
    avg_left_intercept = sum(left_intercept)/len(left_intercept)
    avg_right_intercept = sum(right_intercept)/len(right_intercept)


    x_bottom_left = (540 - avg_left_intercept)/avg_left_slope
    x_bottom_right = (540 - avg_right_intercept)/avg_right_slope
    x_horizon_left = (270 - avg_left_intercept)/avg_left_slope
    x_horizon_right = (270 - avg_right_intercept)/avg_right_slope

    return x_bottom_left, x_bottom_right, x_horizon_left, x_horizon_right


def draw_lane_lines(frame, x_bottom_left, x_bottom_right, x_horizon_right, x_horizon_left):
    frame_cpy = frame.copy()
    cv.line(frame_cpy, (int(x_bottom_left), 540), (int(x_horizon_left), 270), color=(0,0,255), thickness=3)
    cv.line(frame_cpy, (int(x_bottom_right), 540), (int(x_horizon_right), 270), color=(0,0,255), thickness=3)

    cv.imwrite(output_path + "lane_lines.png", frame_cpy)
    cv.imshow("Lined_lanes_Frame", frame_cpy)
    cv.waitKey(0)
    cv.destroyAllWindows()



x_bl, x_br, x_hl, x_hr = extrapolation(detect_lines(apply_mask(get_edges(frame))))
draw_lane_lines(frame, x_bl, x_br, x_hr, x_hl)