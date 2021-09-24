import numpy as np
import cv2


def grayscale(img):
    """Applies the Grayscale transform and image with only one color channel"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """ Applies an image mask to only keep the region of"""
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        fill_color = (255,) * channel_count
    else:
        fill_color = 255

    cv2.fillPoly(mask, vertices, fill_color)

    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """ 'img' is the output of a Canny transform. """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def draw_lines(img, lines, color=[255, 0, 0], thickness=6):
    sizeY = img.shape[0]
    sizeX = img.shape[1]

    pointsLeft = []
    pointsRight = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            # Gets the midpoint of a line
            posX = (x1 + x2) * 0.5
            posY = (y1 + y2) * 0.5

            if posX < sizeX * 0.5:
                pointsLeft.append((posX, posY))
            else:
                pointsRight.append((posX, posY))

    # Get m and b from linear regression
    left_m, left_b = simple_linear_regresion(pointsLeft)
    right_m, right_b = simple_linear_regresion(pointsRight)

    # Define the points of left line     x = (y - b) / m
    left_y1 = int(sizeY)
    left_x1 = int((left_y1 - left_b) / left_m)
    left_y2 = int(sizeY * 0.6)
    left_x2 = int((left_y2 - left_b) / left_m)

    # Define the points of right line     x = (y - b) / m
    right_y1 = int(sizeY)
    right_x1 = int((right_y1 - right_b) / right_m)
    right_y2 = int(sizeY * 0.6)
    right_x2 = int((right_y2 - right_b) / right_m)

    # Draw two lane lines
    cv2.line(img, (left_x1, left_y1), (left_x2, left_y2), color, thickness)
    cv2.line(img, (right_x1, right_y1), (right_x2, right_y2), color, thickness)


def simple_linear_regresion(points):
    n = len(points)
    SumX = 0
    SumY = 0
    SumXY = 0
    SumXX = 0
    m = 0
    b = 0

    for point in points:
        SumX = SumX + point[0]
        SumY = SumY + point[1]
        SumXX = SumXX + (point[0] * point[0])
        SumXY = SumXY + (point[0] * point[1])

    m = ((n * SumXY) - (SumX * SumY)) / ((n * SumXX) - (SumX * SumX))
    b = ((SumY * SumXX) - (SumX * SumXY)) / ((n * SumXX) - (SumX * SumX))

    return m, b


def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """ initial_img * α + img * β + γ """
    return cv2.addWeighted(initial_img, α, img, β, γ)
