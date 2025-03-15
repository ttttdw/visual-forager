import os
import shutil
import random


def split_images(input_folder, output_folder1, output_folder2, split_ratio=0.8):
    # Get the list of image files in the input folder
    image_files = [f for f in os.listdir(
        input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Calculate the number of images to include in each split
    num_images = len(image_files)
    num_images_split1 = int(num_images * split_ratio)

    # Randomly shuffle the list of image files
    random.shuffle(image_files)

    # Create the output folders if they don't exist
    os.makedirs(output_folder1, exist_ok=True)
    os.makedirs(output_folder2, exist_ok=True)

    # Copy images to the first output folder
    for i in range(num_images_split1):
        source_path = os.path.join(input_folder, image_files[i])
        dest_path = os.path.join(output_folder1, image_files[i])
        shutil.copyfile(source_path, dest_path)

    # Copy remaining images to the second output folder
    for i in range(num_images_split1, num_images):
        source_path = os.path.join(input_folder, image_files[i])
        dest_path = os.path.join(output_folder2, image_files[i])
        shutil.copyfile(source_path, dest_path)


if __name__ == '__main__':

    # Example usage
    input_folder = 'visual_foraging_gym/envs/OBJECTSALL'
    output_folder1 = 'visual_foraging_gym/envs/Train'
    output_folder2 = 'visual_foraging_gym/envs/Test'

    split_images(input_folder, output_folder1, output_folder2, split_ratio=0.8)
