import os

def rename_files_in_directory(directory, file_extension):
    """
    Renames all files in a directory by keeping only the first part of the filename
    (up to the first underscore) and removing the rest, while maintaining the original file extension.
    """
    for filename in os.listdir(directory):
        if filename.endswith(file_extension):
            # Split the filename and retain only the first part
            new_name = filename.split('_')[0] + file_extension
            # Construct full old and new file paths
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            # Rename the file
            os.rename(old_path, new_path)
            print(f'Renamed: {old_path} to {new_path}')

# Set the paths to your images and labels directories
images_directory = 'validation_set/images'
labels_directory = 'validation_set/labels'

# Rename files in each directory
rename_files_in_directory(images_directory, '.jpg')  # For image files
rename_files_in_directory(labels_directory, '.txt')  # For label files
