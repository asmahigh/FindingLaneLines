"""Importing packages"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from process import process_image
import os
from moviepy.editor import VideoFileClip

# Test on images
images = os.listdir("test_images/")
if not os.path.isdir("test_images_output"):
    os.mkdir("test_images_output")

for image in images:
    inputImage = mpimg.imread("test_images/" + image)
    outputImage = process_image(inputImage)
    mpimg.imsave("test_images_output/" + image, outputImage)
    plt.imshow(outputImage)
    plt.show()

# Test on Videos
output1 = 'test_videos_output/solidWhiteRight.mp4'
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0, 10)
first_clip = clip1.fl_image(process_image)
first_clip.write_videofile(output1, audio=False)

output2 = 'test_videos_output/solidYellowLeft.mp4'
clip2 = VideoFileClip("test_videos/solidYellowLeft.mp4").subclip(0, 10)
second_clip = clip2.fl_image(process_image)
second_clip.write_videofile(output2, audio=False)
