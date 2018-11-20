
import os
import cv2
import shutil
import random
import xml.etree.ElementTree as ET

def parse_xml_label(label, len_vid):
    """Parse the label file generated automatically by labeling tool vatic.js
    
    Output:
        A list with length of len_vid.
        Each element is a list: (obj_name, xmin, ymin, xmax, ymax)

    """
    tree = ET.parse(label)
    root = tree.getroot()
    objects = root.findall("./object")
    print('{} objects found in lable file {}'.format(len(objects), label))

    result = [[] for i in range(len_vid)]
    
    for obj in objects:
        name = obj.findall('name')[0].text
        obj_type = int(name == 'Goal')     # goal - 1, human - 0
        obj_name = 'goal' if obj_type else 'human'
        print(name, "type=", obj_type)
        
        for poly in obj.findall('polygon'):
            time = int(poly.findall('t')[0].text)
            pts = poly.findall('pt')
            assert len(pts) == 4
            dimension = [pts[0].find('x').text, pts[0].find('y').text,
                pts[2].find('x').text, pts[2].find('y').text]
            result[time].append([obj_name] + dimension)
            # print(result)
            # print('Time', time, 'coord', dimension)
        # assert 0

    return result


def save_voc_xml(objects, shape_im, root_folder, val_ratio=0.2):
    """Save and rename the source image, and save the labeling file in VOC format."""
    train_set = []
    val_set = []

    for idx, frame in enumerate(objects):
        # Skip empty frame
        if len(frame) == 0:
            continue

        base_name = "%06d" % idx

        anno = ET.Element('annotation')
        ET.SubElement(anno, 'folder').text = root_folder
        ET.SubElement(anno, 'filename').text = base_name + ".jpg"
        
        size = ET.SubElement(anno, 'size')
        ET.SubElement(size, 'width').text = str(shape_im[1])
        ET.SubElement(size, 'height').text = str(shape_im[0])
        ET.SubElement(size, 'depth').text = str(shape_im[2])

        for obj in frame:
            obj_node = ET.SubElement(anno, 'object')
            ET.SubElement(obj_node, 'name').text = obj[0]
            ET.SubElement(obj_node, 'difficult').text = '0'
            bndbox = ET.SubElement(obj_node, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = obj[1]
            ET.SubElement(bndbox, 'ymin').text = obj[2]
            ET.SubElement(bndbox, 'xmax').text = obj[3]
            ET.SubElement(bndbox, 'ymax').text = obj[4]
        
        # Copy and rename the source image
        shutil.copy2(os.path.join(root_folder, 'extracted-frames', "{}.jpg".format(idx)), \
            os.path.join(root_folder, "JPEGImages", base_name + ".jpg"))
        # Save the annotation file
        tree = ET.ElementTree(anno)
        tree.write(os.path.join(root_folder,  "Annotations",  base_name + ".xml"))

        # Randomly split into train and val set
        if random.random() < val_ratio:
            val_set.append(base_name)
        else:
            train_set.append(base_name)

        print("Image", base_name + ".jpg")

    return train_set, val_set


def gen_voc_label():
    """
        A reference annotation file.

<annotation>
    <folder>VOC2007</folder>
    <filename>000040.jpg</filename>
    <source>
        <database>The VOC2007 Database</database>
        <annotation>PASCAL VOC2007</annotation>
        <image>flickr</image>
        <flickrid>97167996</flickrid>
    </source>
    <owner>
        <flickrid>ResQgeek</flickrid>
        <name>?</name>
    </owner>
    <size>
        <width>500</width>
        <height>332</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>bird</name>
        <pose>Right</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>1</xmin>
            <ymin>52</ymin>
            <xmax>384</xmax>
            <ymax>290</ymax>
        </bndbox>
    </object>
</annotation>
    """
    pass

def main(label, root_folder):
    # Create folders
    folders = ['Annotations', 'JPEGImages', 'ImageSets', 'ImageSets/Main']
    for f in folders: 
        fold = os.path.join(root_folder, f)
        if not os.path.isdir(fold):
            os.mkdir(fold)
    # Get image info
    for root, folders, files in os.walk(os.path.join(root_folder, 'extracted-frames')):
        break
    len_vid = len(files)
    shape_im = cv2.imread(os.path.join(root, files[0])).shape # (y, x, c)
    print('{} images found, with shape:'.format(len_vid), shape_im)
    # Parse the label file
    objects = parse_xml_label(os.path.join(root_folder, label), len_vid)
    # Save to new label file in VOC format
    train_set, val_set = save_voc_xml(objects, shape_im, root_folder)
    # Save the train and val set id
    with open(os.path.join(root_folder, 'ImageSets', 'Main', 'train.txt'), 'w') as ofile:
        for fr in train_set:
            ofile.write(fr + '\n')
    with open(os.path.join(root_folder, 'ImageSets', 'Main', 'val.txt'), 'w') as ofile:
        for fr in val_set:
            ofile.write(fr + '\n')

if __name__ == '__main__':
    root_folder = 'VID03-goalkeeper'
    label_file = 'output.xml'
    main(label_file, root_folder)