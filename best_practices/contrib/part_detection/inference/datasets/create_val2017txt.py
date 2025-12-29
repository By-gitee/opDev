import os
import argparse

def generate_val_txt(dir_path):
    # Define paths
    val2017_dir = os.path.join(dir_path, "val2017")
    txt_file_path = os.path.join(dir_path, "val2017.txt")

    # Create directories
    os.makedirs(val2017_dir, exist_ok=True)

    # Create val2017.txt and populate with file paths
    with open(txt_file_path, "w") as txt_file:
        for filename in os.listdir(val2017_dir):
            if filename.endswith(".jpg"):
                relative_path = os.path.join("./val2017", filename)
                txt_file.write(relative_path + "\n")

    # Print confirmation
    print("File structure and txt file generated successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a val2017.txt file for COCO dataset")
    parser.add_argument("--dir_path", required=True, help="Path to the dataset root directory")

    args = parser.parse_args()
    generate_val_txt(args.dir_path)
