from PIL import Image, ImageOps
import os
import math

def divide_image(image_path, output_folder):
    # Open the original image
    image = Image.open(image_path)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    width, height = image.size

    # Calculate number of divisions
    x_divs = max(1, math.ceil(width / 512))
    y_divs = max(1, math.ceil(height / 512))

    # Pad the image if necessary
    if width < 512 or height < 512:
        pad_width = max(0, 512 - width)
        pad_height = max(0, 512 - height)
        padding = (0, 0, pad_width, pad_height)
        image = ImageOps.expand(image, padding, fill=(0, 0, 0))
        width, height = image.size

    # Calculate overlap if needed
    x_overlap = (x_divs * 512 - width) // (x_divs - 1) if x_divs > 1 else 0
    y_overlap = (y_divs * 512 - height) // (y_divs - 1) if y_divs > 1 else 0

    count = 1
    for y in range(y_divs):
        for x in range(x_divs):
            left = x * 512 - x * x_overlap
            upper = y * 512 - y * y_overlap
            right = min(left + 512, width)
            lower = min(upper + 512, height)

            # Crop the image
            cropped_image = image.crop((left, upper, right, lower))

            # Ensure the cropped image is exactly 512x512 with padding if needed
            if cropped_image.size != (512, 512):
                cropped_image = ImageOps.pad(cropped_image, (512, 512), color=(0, 0, 0))

            # Save the cropped image
            output_path = os.path.join(output_folder, f"{image_name}_{count}_of_{x_divs * y_divs}.jpeg")
            cropped_image.save(output_path, format='JPEG')
            count += 1

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.jpg'):
            image_path = os.path.join(input_folder, filename)
            divide_image(image_path, output_folder)

if __name__ == "__main__":
    input_folder = "data_formatted/raw"
    output_folder = "data_formatted/images_split_jpg"
    process_folder(input_folder, output_folder)