import os
import shutil
from pathlib import Path
import cv2
import numpy as np
import pytesseract

# Configure the path to tesseract executable if necessary
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Source directory
source_dir = Path(r"C:\Users\aksha\Downloads\Target\Target")

# Destination directories
marked_dir = source_dir / "images marked"
non_marked_dir = source_dir / "images non marked"

# Create destination directories if they don't exist
marked_dir.mkdir(exist_ok=True)
non_marked_dir.mkdir(exist_ok=True)

# Define image extensions
image_extensions = ['.jpg', '.jpeg', '.tif', '.tiff', '.png']   

# Define color ranges in HSV for red, yellow, blue, and green
color_ranges = {
    'red1': ((0, 70, 50), (10, 255, 255)),
    'red2': ((170, 70, 50), (180, 255, 255)),
    'yellow': ((20, 70, 50), (30, 255, 255)),
    'blue': ((100, 70, 50), (130, 255, 255)),
    'green': ((40, 70, 50), (70, 255, 255)),
}

def has_colored_text(image):
    # Convert image to grayscale for text detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use pytesseract to detect text boxes
    boxes = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

    n_boxes = len(boxes['level'])
    has_colored_text_flag = False

    for i in range(n_boxes):
        (x, y, w, h) = (boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i])

        # Extract the text region from the image
        text_region = image[y:y+h, x:x+w]

        # Convert to HSV color space
        hsv = cv2.cvtColor(text_region, cv2.COLOR_BGR2HSV)

        # Check for each color
        mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for color in color_ranges:
            if color.startswith('red'):
                lower, upper = color_ranges[color]
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                mask_total = cv2.bitwise_or(mask_total, mask)
            else:
                lower, upper = color_ranges[color]
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                mask_total = cv2.bitwise_or(mask_total, mask)

        # If enough pixels in the text region match the color, mark as having colored text
        if cv2.countNonZero(mask_total) > 0.2 * w * h:
            has_colored_text_flag = True
            break

    return has_colored_text_flag

# Loop through all files recursively
for file_path in source_dir.rglob('*'):
    if file_path.is_file() and file_path.suffix.lower() in image_extensions:
        # Read the image
        image = cv2.imread(str(file_path))

        if image is None:
            continue  # Skip files that cannot be read as images

        if has_colored_text(image):
            dest_file = marked_dir / file_path.name
        else:
            dest_file = non_marked_dir / file_path.name

        # Handle if file already exists
        if dest_file.exists():
            counter = 1
            new_name = dest_file.parent / f"{file_path.stem}_{counter}{file_path.suffix}"
            while new_name.exists():
                counter += 1
                new_name = dest_file.parent / f"{file_path.stem}_{counter}{file_path.suffix}"
            dest_file = new_name

        # Move the file
        shutil.move(str(file_path), str(dest_file))
