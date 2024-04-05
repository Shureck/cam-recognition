import os

def get_folders(directory):
    folders = [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]
    return folders

# directory_path = 'photos'
# folders = get_folders(directory_path)
# print("Folders in", directory_path, ":", folders)