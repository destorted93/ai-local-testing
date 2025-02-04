import os
import random
import string

def create_random_folder_structure(base_path, depth=0):
    if depth > 5:
        return

    # Create a random folder name
    folder_name = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    folder_path = os.path.join(base_path, folder_name)

    # Create the folder
    os.makedirs(folder_path, exist_ok=True)

    # Recursively create subfolders
    num_subfolders = random.randint(5, 10)
    for _ in range(num_subfolders):
        create_random_folder_structure(folder_path, depth + 1)

# Start the structure from the base folder
base_folder = 'fake_bin'
create_random_folder_structure(base_folder)

print(f"Folder structure created in {base_folder}")