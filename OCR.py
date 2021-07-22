"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""

import matplotlib
import argparse
import json
import os
import glob
import cv2
import numpy as np


def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args


def enrollment(character_list):
    """ Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    
    
    
    """
    
    cropped_list_char = list()
    cropped_list_char_val = list()
    for alphabet, alphabet_img in character_list:
        cut, mat = cv2.threshold(alphabet_img, 150, 255, cv2.THRESH_BINARY_INV) 
        x = np.where(mat == 255)
        x1 = np.min(x[0])
        x2 = np.min(x[1])
        x3 = np.max(x[0])
        x4 = np.max(x[1])
        a = mat[x1:x3, x2:x4]
        a = cv2.resize(a, (25,25))
        a = cv2.blur(a, (5,5))
        cropped_list_char.append([a])
        cropped_list_char_val.append(alphabet)
    return cropped_list_char, cropped_list_char_val
    
    
    
    # TODO: Step 1 : Your Enrollment code should go here.
    #raise NotImplementedError

def prior_neighbours(labeled_image, i, j):
    if (i == 0):
        top_label = 0;
    else:
        top_label = labeled_image[i-1][j]
    
    if (j == 0):
        left_label = 0;
    else:
        left_label = labeled_image[i][j-1]
    
    return left_label, top_label

def getParent(labelDict,label):
    
    connectedComponents = list(labelDict[label])
    i = 0
    while i < len(connectedComponents):
        for (key, value) in labelDict.items():
            if connectedComponents[i] in value:
                connectedComponents.extend(list(value))
                connectedComponents = list(dict.fromkeys(connectedComponents))
        i+=1        
    return min(connectedComponents)  

def connected_components(image): 
    labelled_image = np.zeros((image.shape[0], image.shape[1]))
    label_val = 1
    
    LabelDict = dict()
    currentLabel = 0
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if(image[i][j] > 200):
                
                A = prior_neighbours(labelled_image, i,j)
                if A == (0,0):
                    currentLabel = label_val                    
                    labelled_image[i][j] = currentLabel
                    LabelDict[currentLabel] = {currentLabel}
                    label_val = label_val + 1
                else:
                    currentLabel = min(A) if min(A) != 0 else max(A)
                    labelled_image[i][j] = currentLabel
                    if(min(A) != 0):
                        LabelDict[currentLabel] = LabelDict[currentLabel].union(set(A))
                    else:
                        LabelDict[currentLabel] = LabelDict[currentLabel].union({max(A)})

    for i in range(labelled_image.shape[0]):
        for j in range(labelled_image.shape[1]):
            if labelled_image[i][j] == 0: 
                continue
            labelled_image[i][j] = getParent(LabelDict,labelled_image[i][j])


    return labelled_image

def detection(img):
    """ 
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    cut, new_img = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY_INV)
    
    labelled_image = connected_components(new_img)
    
    # label
    connectedArray = list()
    for i in range(labelled_image.shape[0]):
            for j in range(labelled_image.shape[1]):
                if labelled_image[i][j] == 0:
                    continue
                lableFound = False
                for prevLable in connectedArray:
                    if prevLable == labelled_image[i][j]:
                        lableFound = True
                        break
                if lableFound == False:
                    connectedArray.append(labelled_image[i][j])
    
    
    
    #all pixel values
    connected_locations = list()
    for component in connectedArray:
        temp_list = list()
        for i in range(labelled_image.shape[0]):
                for j in range(labelled_image.shape[1]):
                    if labelled_image[i][j] == component:
                        pixel_pos = [i,j]
                        temp_list.append(pixel_pos)
        connected_locations.append(temp_list)
        
    
    
    list_components = list()
    width_height = list()
    for i in connectedArray:
        x = np.where(labelled_image == i)
        x1 = np.min(x[0])
        x2 = np.min(x[1])
        x3 = np.max(x[0])
        x4 = np.max(x[1])
        height = (x3 - x1)
        width = (x4-x2)
        # if width == 0 or height ==0:
        #     continue
        width_height.append([int(width), int(height)])
        a=labelled_image[x1:x3, x2:x4]
        a = cv2.resize(a, (25,25))
        cut, a = cv2.threshold(a, 0.5, 255, cv2.THRESH_BINARY)
        a = cv2.blur(a, (5,5))
        list_components.append(a)

    final_box = list()
    for i in range(len(connected_locations)):
        a = connected_locations[i][0]
        b = list()
        b.extend(a)
        b.extend(width_height[i])
        final_box.append(b)

    return list_components,final_box
    # TODO: Step 2 : Your Detection code should go here.
    #raise NotImplementedError

def recognition(list_components, cropped_list_char, final_box, cropped_list_char_name):
    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 3 : Your Recognition code should go here.
    recognition_list = list()
    for i in range(len(list_components)):
        ssd_list = list()
        ssd_bool = False
        min_val = 16
        min_val2 = 0
        for j in range(len(cropped_list_char)):
            ssd_val = np.sum(((list_components[i]) - (cropped_list_char[j]))**2)
            ssd_val=(float(ssd_val))/100000
            ssd_list.append(ssd_val)
            
            if min(ssd_list)<min_val:    
                min_val = min(ssd_list)            
                ssd_bool = True
                min_val2 =j
        if ssd_bool == False:
            recognition_list.append({"bbox":final_box[i],"name":"UNKNOWN"})
        else:
            recognition_list.append({"bbox":final_box[i],"name":cropped_list_char_name[min_val2]})
        
        
    count =0

    for i in range(len(list_components)):
        #print(i)
        for j in range(len(cropped_list_char)):
            #print(j)
            ssd_val = np.sum(((list_components[i]) - (cropped_list_char[j]))**2)
            ssd_val=(float(ssd_val))/100000

            if ssd_val<20:
                count+=1
                     
    return recognition_list
    
  
    
    
    #raise NotImplementedError



def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # TODO Add your code here. Do not modify the return and input arguments
    
    
    
    
    features, feature_vals = enrollment(characters)
    components, bbox = detection(test_img)
    result_final = recognition(components,features,bbox, feature_vals)
    

    
    
    return result_final

    #raise NotImplementedError



def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = coordinates
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()
    
    characters = []
    all_character_imgs = glob.glob(args.character_folder_path+ "/*")
    
    for each_character in all_character_imgs :
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])


    test_img = read_image(args.test_img)
    sh = test_img
    results = ocr(test_img, characters)
    save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()

