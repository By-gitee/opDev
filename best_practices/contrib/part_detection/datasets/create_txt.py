import os
import argparse

def create_txt(dataset_name, image_path, output_path):
    path = "./" + dataset_name + "/" + image_path
    output_path = "./" + dataset_name+"/" + output_path
    with open(output_path, 'w') as file:
        for filename in os.listdir(path):
            if filename.endswith('.jpg'):
                file.write(os.path.join("./", image_path, filename) + '\n')

def main(dataset_name):
    dataset_paths = [
        {"image_path": 'images/train2017/', "output_file": f'train2017.txt'},
        {"image_path": 'images/val2017/', "output_file": f'val2017.txt'}
        # Add more dataset paths here if needed
    ]

    for dataset in dataset_paths:
        create_txt(dataset_name, dataset["image_path"], dataset["output_file"])
        print(f"Created {dataset['output_file']} from {dataset['image_path']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='mycoco1')
    opt = parser.parse_args()
    
    main(opt.dataset_name)
