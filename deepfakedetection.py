import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from mtcnn.mtcnn import MTCNN
from skimage import measure
import os

# Function to extract frames from multiple videos
def extract_multiple_videos(input_filenames, image_path_infile):
    """Extract video files into sequence of images."""
    i = 1  # Counter for image filenames
    
    # Iterate through the list of video filenames
    for filename in input_filenames:
        cap = cv2.VideoCapture(filename)
        
        if not cap.isOpened():
            print(f"Error opening file {filename}")
            continue
        
        # Iterate through video frames
        while True:
            ret, frame = cap.read()  # Read frame from video
            
            if ret:
                # Write frame to JPEG file
                cv2.imwrite(os.path.join(image_path_infile, f'{i}.jpg'), frame)
                # Optionally display the frame
                # cv2.imshow('frame', frame)
                i += 1  # Advance file counter
            else:
                break  # Break the loop when no more frames are read
        
        cap.release()
        cv2.waitKey(50)  # Wait 50ms between frames

# Example usage
fake_video_name = ['fake_video1.avi', 'fake_video2.avi']
fake_image_path_for_frame = 'path_to_fake_images'
real_video_name = ['real_video1.avi', 'real_video2.avi']
real_image_path_for_frame = 'path_to_real_images'

extract_multiple_videos(fake_video_name, fake_image_path_for_frame)
extract_multiple_videos(real_video_name, real_image_path_for_frame)

# Function to calculate Mean Squared Error between two images
def mse(imageA, imageB):
    """Calculate the Mean Squared Error between two images."""
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err  # Return the MSE, lower error indicates more similarity

# Function to compare two images using MSE and SSIM
def compare_images(imageA, imageB, title):
    """Compare two images and display their MSE and SSIM."""
    m = mse(imageA, imageB)
    s = measure.compare_ssim(imageA, imageB)
    
    # Setup the figure for displaying images
    fig = plt.figure(title)
    plt.suptitle(f"MSE: {m:.2f}, SSIM: {s:.2f}")
    
    # Display the first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")
    
    # Display the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")
    
    # Show the images
    plt.show()
