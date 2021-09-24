import numpy as np
from helpers import grayscale, gaussian_blur, canny, region_of_interest, hough_lines, weighted_img


def process_image(image):
    ysize = image.shape[0]
    xsize = image.shape[1]

    # define region of interest
    vertices = np.array([[(xsize * 0.10, ysize * 0.90),
                          (xsize * 0.46, ysize * 0.60),
                          (xsize * 0.54, ysize * 0.60),
                          (xsize * 0.90, ysize * 0.90)]], dtype=np.int32)

    gray = grayscale(image)  # (source_img)
    blur = gaussian_blur(gray, 9)  # (gray_img, kernel size)
    edges = canny(blur, 100, 200)  # (blurred_img, low_threshold, high_threshold)
    masked_edges = region_of_interest(edges, vertices)  # (edges_img, vertices)
    line_img = hough_lines(masked_edges, 1, np.pi / 180, 10, 5, 5) # (masked_edges_img, rho, theta, threshold,
    # min_line_len, max_line_gap)
    weighted = weighted_img(line_img, image, 0.8, 1.0, 0.0)  # (lines_image, source_image, α , β , γ)

    return weighted

