import os
import re

# make sure that the cwd() in the beginning is the location of the python script (so that every path makes sense)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

IN_FILE = 'result.txt'

# change directory to the one with the files to be changed
parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
parent_path = os.path.abspath(os.path.join(parent_path, os.pardir))
DR_PATH = os.path.join(parent_path, 'input','detection-results')
#print(DR_PATH)
os.chdir(DR_PATH)

SEPARATOR_KEY = 'Enter Image Path:'
IMG_FORMAT = '.jpg'
result_format = '%'

# outfile = None
fo = open(os.path.join(DR_PATH, 'result.txt'), "r")
alllines = fo.readlines()  #依次读取每行  
   # alllines = alllines.strip()    #去掉每行头尾空白  
# 关闭文件
fo.close()
for line in alllines:
    #if SEPARATOR_KEY in line:
    if IMG_FORMAT in line:
    #    if IMG_FORMAT not in line:
    #        break
        # get text between two substrings (SEPARATOR_KEY and IMG_FORMAT)
        #image_path = re.search(SEPARATOR_KEY + '(.*)' + IMG_FORMAT, line)
        # get the image name (the final component of a image_path)
        # e.g., from 'data/horses_1' to 'horses_1'
        #image_name = os.path.basename(image_path.group(1))
        image_path = (line.split(':', 1))[0]
        image_name = (image_path.split('/',2))[2]
        image_name = (image_name.split('.',1))[0]
        
        
        # close the previous file
        #if outfile is not None:
        #    outfile.close()
        # open a new file
        
    # elif outfile is not None:
    elif result_format in line:
        # split line on first occurrence of the character ':' and '%'
        outfile = open(os.path.join(DR_PATH, image_name + '.txt'), 'w')
        class_name, info = line.split(':', 1)
        confidence, bbox = info.split('%', 1)
        # get all the coordinates of the bounding box
        bbox = bbox.replace(')','') # remove the character ')'
        # go through each of the parts of the string and check if it is a digit
        left, top, width, height = [int(s) for s in bbox.split() if s.lstrip('-').isdigit()]
        right = left + width
        bottom = top + height
        outfile.write("{} {} {} {} {} {}\n".format(class_name, float(confidence)/100, left, top, right, bottom))
        outfile.close()
