import os
import xml.etree.ElementTree as ET

def voc_annotation_to_yolo(voc_dir):
    # path
    anno_dir = os.path.join(voc_dir, 'Annotations')
    label_file = os.path.join(voc_dir, 'classes.txt')

    # yolo path
    yolo_dir = os.path.join(voc_dir, 'yolo')
    if not os.path.exists(yolo_dir):
        os.mkdir(yolo_dir)

    with open(label_file, 'r') as f:
        labels = f.read().splitlines()
        
    voc_label_to_yolo = {label: i for i, label in enumerate(labels)}

    for anno_file in os.listdir(anno_dir):
        tree = ET.parse(os.path.join(anno_dir, anno_file))
        root = tree.getroot()
        
        img_file = root.find('filename').text
        img_w = int(root.find('size/width').text)
        img_h = int(root.find('size/height').text)
        
        yolo_file = os.path.join(yolo_dir, os.path.splitext(anno_file)[0] + '.txt')
        with open(yolo_file, 'w') as f:
            for obj in root.findall('object'):
                obj_cls = obj.find('name').text
                x_min = int(obj.find('bndbox/xmin').text)
                y_min = int(obj.find('bndbox/ymin').text)
                x_max = int(obj.find('bndbox/xmax').text)
                y_max = int(obj.find('bndbox/ymax').text)
                
                obj_cls_id = voc_label_to_yolo[obj_cls]
                
                obj_x = (x_min + x_max) / 2 / img_w
                obj_y = (y_min + y_max) / 2 / img_h
                obj_w = (x_max - x_min) / img_w
                obj_h = (y_max - y_min) / img_h
                
                f.write(f'{obj_cls_id} {obj_x} {obj_y} {obj_w} {obj_h}\n')
                
if __name__=="__main__":

    voc_dir = './VOCdevkit/VOC2012'

    voc_annotation_to_yolo(voc_dir)
    
    print("VOC格式转化成yolo格式，转化完成!")