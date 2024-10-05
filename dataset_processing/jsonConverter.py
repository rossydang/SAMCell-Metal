import os
import cv2
import json
import uuid
import numpy as np

# Set paths for input and output folders
input_folder_npy = 'data_formatted/annotations_npy'
output_folder = 'data_formatted/annotations_darwin_json'
input_folder_jpg = 'data_formatted/images_resized_jpg'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop over all .npy files in the input folder
for npy_file in os.listdir(input_folder_npy):
    print(f"Processing {npy_file}")
    if npy_file.endswith('.npy'):
        # Extract the base name without extension
        base_name = os.path.splitext(npy_file)[0]

        # Load the mask
        mask_path = os.path.join(input_folder_npy, npy_file)
        mask = np.load(mask_path)
        print(f"Processing {base_name}, mask shape: {mask.shape}, dtype: {mask.dtype}")

        # Load the corresponding image to get dimensions
        image_path = os.path.join(input_folder_jpg, f"{base_name}.jpg")
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"The image '{base_name}.jpg' was not found.")
        height, width = image.shape[:2]

        # Get the unique labels in the mask (excluding background label 0)
        labels = np.unique(mask)
        labels = labels[labels != 0]

        annotations = []

        for label in labels:
            # Create a binary mask for the current label
            binary_mask = np.uint8((mask == label) * 255)
            if len(binary_mask.shape) != 2:
                binary_mask = np.squeeze(binary_mask)
            print(f"binary_mask shape: {binary_mask.shape}, dtype: {binary_mask.dtype}")

            if binary_mask.ndim > 2:
                binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)

            # Find contours for the current label
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            print(f"Found {len(contours)} contours for label {label}")

            paths_and_boxes = []
            x_min, y_min, x_max, y_max = width, height, 0, 0

            for cnt in contours:
                cnt = cnt.squeeze()
                if cnt.ndim != 2 or cnt.shape[1] != 2:
                    continue  # Skip invalid contours

                # Update bounding box coordinates
                x, y, w, h = cv2.boundingRect(cnt)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)

                bounding_box = {
                    'h': float(y_max - y_min),
                    'w': float(x_max - x_min),
                    'x': float(x_min),
                    'y': float(y_min)
                }

                # Convert contour to a list of points
                path_and_box = (bounding_box, [{'x': float(point[0]), 'y': float(point[1])} for point in cnt])
                paths_and_boxes.append(path_and_box)

            if not paths_and_boxes:
                continue  # Skip if no valid contours found

            # # Define the bounding box
            # bounding_box = {
            #     'h': float(y_max - y_min),
            #     'w': float(x_max - x_min),
            #     'x': float(x_min),
            #     'y': float(y_min)
            # }

            for path_and_box in paths_and_boxes:
                # Create the annotation
                annotation = {
                    'bounding_box': path_and_box[0],
                    'id': str(uuid.uuid4()),
                    'name': 'grain',
                    'polygon': {
                        'paths': path_and_box[1]
                    },
                    'slot_names': ["0"],
                    'properties': []
                }

                annotations.append(annotation)

        # Build the final JSON structure
        json_data = {
            "version": "2.0",
            "schema_ref": "https://darwin-public.s3.eu-west-1.amazonaws.com/darwin_json/2.0/schema.json",
            "item": {
                "name": f"{base_name}.jpg",
                "path": "/",
                "slots": [
                    {
                        "type": "image",
                        "slot_name": "0",
                        "width": width,
                        "height": height,
                        "thumbnail_url": "",
                        "source_files": [
                            {
                                "file_name": f"{base_name}.jpg",
                                "url": ""
                            }
                        ]
                    }
                ]
            },
            "annotations": annotations
        }

        # Save the JSON data to a file
        output_json_path = os.path.join(output_folder, f"{base_name}.json")
        with open(output_json_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        print(f"Saved annotations to {output_json_path}")
