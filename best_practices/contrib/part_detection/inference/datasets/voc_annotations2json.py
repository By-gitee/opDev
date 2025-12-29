import xml.etree.ElementTree as ET
import json
import os


def voc_annotations2json(category_mapping, xml_dir, output_path):
    # Create an empty result list
    result = []

    # Iterate over all XML files in the directory
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith(".xml"):
            xml_path = os.path.join(xml_dir, xml_file)

            # Parse the XML file
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Loop through objects in the XML
            for obj in root.findall("object"):
                image_id = int(root.find("filename").text.split(".")[0])  # Get image_id

                category_name = obj.find("name").text  # Get category name
                category_id = category_mapping.get(category_name, 0)  # Find category ID, default to 0 if not found

                xmin = float(obj.find("bndbox/xmin").text)
                ymin = float(obj.find("bndbox/ymin").text)
                xmax = float(obj.find("bndbox/xmax").text)
                ymax = float(obj.find("bndbox/ymax").text)

                # Calculate bbox coordinates
                bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

                result.append({
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox
                })

    # Write the result to a unified JSON file
    with open(output_path, "w") as json_file:
        json.dump(result, json_file, indent=2)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='YOLOv5 datasets setting')    
    parser.add_argument('--xml_dir', type=str, default="./multi_imgs/Annotations", help='')
    parser.add_argument('--output_path', type=str, default="./multi_imgs/instances_val2017.json", help='')
    opt = parser.parse_args()
    
    # Create a mapping dictionary to map category names to their respective IDs
    # Case-sensitive
    category_mapping = {
        "Plug_defect": 1,
        "Screw_defect": 2,
        "fan_defect": 3,
        "fan_screw_missing": 4,
        "Screw_tilt_defect": 5,
        "Screw_type_defect_1": 6,
        "Screw_type_defect_2": 7
    }

    # Specify the directory containing XML files
    # xml_dir = "./multi_imgs/annotations"
    # output_path = "./multi_imgs/instances_val2017.json"
    
    voc_annotations2json(category_mapping, opt.xml_dir, opt.output_path)
    print("annotations to json finished")