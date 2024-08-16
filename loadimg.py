import cv2
import glob

# Directory where your images are stored
def loadimage():
    images_dir = './cameracallibration/'

    # Pattern to match filenames (assuming sequential numbering from PIC_0263.jpg to PIC_0283.jpg)
    file_pattern = 'PIC_0*.jpg'  # Adjust pattern based on actual filenames

    # Get list of image file paths matching the pattern
    image_files = sorted(glob.glob(images_dir + file_pattern))

    # Initialize an empty list to store loaded images
    images = []

    # Loop through each image file path and load the image
    for image_file in image_files:
        img = cv2.imread(image_file)
        if img is not None:
            images.append(img)
        else:
            print(f"Failed to load image: {image_file}")

    # Check if images were successfully loaded
    if len(images) == 0:
        print("No images loaded.")
    else:
        print()

    # Now 'images' list contains all loaded images from PIC_0263.jpg to PIC_0283.jpg

    return images


