import cv2
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import os
from PIL import Image
import shutil

#Define a contrast control function
def contrast_control_function(pixel_value, contrast_value):
    # Implement contrast control logic here (linear scaling)
    return pixel_value * contrast_value / 255

def clahe_preprocess(sourceFilePath,destFilePath,size=224):
    print(sourceFilePath," : ",destFilePath)

    img = cv2.imread(sourceFilePath, 0)  # read image from directory
    img = cv2.resize(img, (size, size))

    # Apply Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_image = clahe.apply(img)

    # Create Antecedent and Consequent objects
    input_image = ctrl.Antecedent(np.arange(0, 256, 1), 'input_image')
    output_image = ctrl.Consequent(np.arange(0, 256, 1), 'output_image')

    # Define fuzzy sets and membership functions for input (original) image
    input_image['low'] = fuzz.trimf(input_image.universe, [0, 0, 127])
    input_image['medium'] = fuzz.trimf(input_image.universe, [0, 255, 255])
    input_image['high'] = fuzz.trimf(input_image.universe, [127, 255, 255])

    # Define fuzzy sets and membership functions for output (normalized) image
    output_image['low'] = fuzz.trimf(output_image.universe, [0, 0, 127])
    output_image['medium'] = fuzz.trimf(output_image.universe, [0, 255, 255])
    output_image['high'] = fuzz.trimf(output_image.universe, [127, 255, 255])

    # Define fuzzy sets and membership functions for contrast control
    contrast_control = ctrl.Antecedent(np.arange(0, 256, 1), 'contrast_control')
    contrast_control['low'] = fuzz.trimf(contrast_control.universe, [0, 0, 127])
    contrast_control['medium'] = fuzz.trimf(contrast_control.universe, [0, 255, 255])
    contrast_control['high'] = fuzz.trimf(contrast_control.universe, [127, 255, 255])

    # Define fuzzy rules for contrast enhancement
    rule1 = ctrl.Rule(input_image['low'] & contrast_control['low'], output_image['low'])
    rule2 = ctrl.Rule(input_image['medium'] & contrast_control['medium'], output_image['medium'])
    rule3 = ctrl.Rule(input_image['high'] & contrast_control['high'], output_image['high'])

    # Create a Fuzzy Control System
    image_fche = ctrl.ControlSystem([rule1, rule2, rule3])

    # Create a Fuzzy Control System simulation
    image_fche_sim = ctrl.ControlSystemSimulation(image_fche)

    # Apply FCHE to the histogram-equalized image
    output_image = np.zeros_like(equalized_image)

    # Define contrast control value
    contrast_value = 100

    for i in range(equalized_image.shape[0]):
        for j in range(equalized_image.shape[1]):
            # Apply contrast control based on pixel intensity
            contrast_control_value = contrast_control_function(equalized_image[i, j], contrast_value)
            # Apply fuzzy control
            image_fche_sim.input['input_image'] = equalized_image[i, j]
            image_fche_sim.input['contrast_control'] = contrast_control_value
            image_fche_sim.compute()
            output_image[i, j] = image_fche_sim.output['output_image']

    # Convert the output to the uint8 data type
    output_image = np.uint8(output_image)
    cv2.imwrite(destFilePath, output_image)  # save output to new paths
def process_images(source_folder, destination_folder):
    # Iterate through the source folder and its subfolders
    for foldername, subfolders, filenames in os.walk(source_folder):
        # Create corresponding folder structure in the destination path
        relative_path = os.path.relpath(foldername, source_folder)
        destination_subfolder = os.path.join(destination_folder, relative_path)
        os.makedirs(destination_subfolder, exist_ok=True)

        # Process files in the current folder
        for filename in filenames:
            source_file_path = os.path.join(foldername, filename)
            destination_file_path = os.path.join(destination_subfolder, filename)

            # Check if the file is an image
            try:
                with Image.open(source_file_path) as img:
                    # Print the image name
                    print("Image found:", filename)

                    # Maninpulate CLAHE Preprocessing
                    clahe_preprocess(source_file_path, destination_file_path)
            except (IOError, OSError):
                # The file is not a valid image file
                print("Not an image:", filename)

                # Move the non-image file to the destination folder
                shutil.move(source_file_path, destination_file_path)


source_folder = 'D:/MURA-v1.1'  # Replace this with the source folder path
destination_folder = 'D:/Mura HE V1.1'  # Replace this with the destination folder path
process_images(source_folder, destination_folder)

