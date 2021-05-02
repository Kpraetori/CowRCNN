import numpy as np
import cv2
import os
from pathlib import Path
from typing import List, Tuple

class BBconversion:
    """Dataloader currently the directory path and augmentation boleen. 
    Images are pulled and annotations are taken from the text file and converted from YOLO to xyxy. Return is a tuple with a list and np array. This makes the data ready for Selective Segmentation."""
    def __init__(self, directory: str):
        self.directory = directory
        self.annotation_path = list(Path(self.directory).glob('*.txt')) # path for annotations

    @staticmethod # used for functions with no class parameter like helper and maths functions
    def _format_annotations(annotation_path: str): # _ first because function kept within
        with open(annotation_path, 'r') as ann_file:
            annotation_strings = [line for line in ann_file] # for each line in the txt file, make a string
        # Now take the strings, pull each value, and add them to a list
        formatted_annotations = []
        for ann_strings in annotation_strings:
            c, x, y, w, h = ann_strings.split(' ') 
            c = int(c)
            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)
            formatted_annotations.append([c, x, y, w, h])
            # print(f'The YOLO format is: {formatted_annotations}')
        return formatted_annotations

    def _open_image(self, annotation_path: str) -> np.array:
        path_no_ext = os.path.splitext(annotation_path)[0] # split the annotation path and select not the extension
        file_name = os.path.basename(path_no_ext)
        # print(file_name)
        dir_path = Path(self.directory).glob(f'{file_name}*')
        image_path = [d_path for d_path in dir_path if not d_path.name.endswith('.txt')]
        if len(image_path) > 0:
            image_path = image_path[0]
        im = cv2.imread(str(image_path))
        # cv2.imshow('Printed Image', im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return im

    @staticmethod
    def convert_yolo_to_xyxy(anns: List, img: np.array):
        xyxy_anns = []
        dh, dw, _ = img.shape
        for bb in anns:
            _, x, y, w, h = bb    
            x1 = int((x - w / 2) * dw)
            x2 = int((x + w / 2) * dw)
            y1 = int((y - h / 2) * dh)
            y2 = int((y + h / 2) * dh)

            # x1 = (x * dw) - (w * dw) / 2
            # x2 = (x * dw) + (w * dw) / 2
            # y1 = (y * dh) - (h * dh) / 2
            # y2 = (y * dh) + (h * dh) / 2
            xyxy_anns.append([x1, y1, x2, y2])
        # print(f'The xy coordinates are: {xyxy_anns}')        
        return xyxy_anns

    def __len__(self): # having a len fun is required. 
        return len(self.annotation_path)

    # idx is for index. the -> shows what format I think...
    def __getitem__(self, idx: int) -> Tuple[List, np.array]: 
        yoloann = self._format_annotations(self.annotation_path[idx])
        img = self._open_image(self.annotation_path[idx])
        ann = self.convert_yolo_to_xyxy(yoloann, img)
        return ann, img


# ra = [0., 0., 7., 7.] # ground truth
# rb = [4., 4., 11., 11.] # proposed region
# # union of these should be 0.25


# MY IOU
def calculate_IOU(coordinatesA, coordinatesB):
    """Ground truth is coordinates A, proposed region is coordinates B. Calculate intersection over union with inputs of ground truth and predicted bb provided in any order. Assumes both coordinates inputs are in x1y1x2y2 list format."""
    # Use largest and smallest (max and min) to get intersection coord
    xLeft = max(coordinatesA[0], coordinatesB[0])
    xRight = min(coordinatesA[2], coordinatesB[2])
    yTop = max(coordinatesA[1], coordinatesB[1])
    yBottom = min(coordinatesA[3], coordinatesB[3])

    # Can these be evaluated?
    if xRight >= xLeft and yBottom >= yTop:
        # area of the rectangle of intersection
        intersection = (xRight - xLeft + 1) * (yBottom - yTop + 1)
        # print(f'Intersection is {intersection}')

        # Area of both boxes and get union
        areaA = (coordinatesA[2] - coordinatesA[0] + 1) * (coordinatesA[3] - coordinatesA[1] + 1)
        # print(f'Area of A is {areaA}')
        areaB = (coordinatesB[2] - coordinatesB[0] + 1) * (coordinatesB[3] - coordinatesB[1] + 1)
        # print(f'Area of B is {areaB}')
        union = float(areaA + areaB - intersection)
        # print(f'Union is {union}')

        iou = intersection/union
    else:
        iou = 0.0
    return iou

# reply = calculate_IOU(ra,rb)
# print(f"IOU is {reply}")

# this prints the ss results on the cow image to see that it works
# only shows 30 SS boxes at a time
def visualiseSS(yourImage, boxesVariable):
    for region in range(0, len(boxesVariable), 30):
        copy = yourImage.copy()
        for (x,y,w,h) in boxesVariable[region: region + 30]:
            cv2.rectangle(copy, (x,y), (x + w, y + h), (0,255,100), 2)
        cv2.imshow("Sample Regions on Image", copy)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
    	    break

# this prints the bb on the cow image to see if it works
def cowfaceBB(img, ann):
        for region in range(0, len(ann), 30):
            copy = img.copy()
        for (x1,y1,x2,y2) in ann[region: region + 30]:
            cv2.rectangle(copy, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,100), 2)
        cv2.imshow("BB Regions on Image", copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# For testing new function
# CounterA = 2
# CounterB = 4
# Data = cv2.imread('data/1.mp4_frame_2790.jpg')


# cowFacePath = 'results/Cowfaces'
# backgroundPath = 'results/Background'

# code requires a face count and background count variable
def save_ROI(RegionData, counterName, savePath: str, Face: bool):
    if Face is True:
        imageName = "cowface{}.png".format(counterName)
    else:
        imageName = "background{}.png".format(counterName)
    finalPath = os.path.sep.join([savePath,imageName])
    resize_image = cv2.resize(RegionData, (224,224), interpolation=cv2.INTER_CUBIC)
    return cv2.imwrite(finalPath, resize_image) and print(f'{imageName} is saved')


# testA = save_ROI(Data, CounterA, cowFacePath, Face=True)
# testB = save_ROI(Data, CounterB, backgroundPath, Face=False)


def collect_test_data(filePath, destinationPath, percent: float):
    """Filepath is where the data is currently stored, the destination is the test folder, and the percent is a float of how much of the data you want to be in the test set. Example 0.05 for 5%."""
    # create a list of files in the folder
    files = [f for f in os.listdir(filePath) if os.path.isfile(os.path.join(filePath, f))]
    # shuffle the list
    random.shuffle(files)
    #find out how many files we need to grab
    selection = int(len(files)*percent)
    # make a list containing up to the selction number of files
    selected_list = files[:selection]
    # loop through list and mmove the files to a new folder
    for file in selected_list:
        shutil.move(os.path.join(filePath, file), destinationPath)
        print(file)

# starting_folder = "OneDriveSample"
# ending_folder = "resultsForTesting/Background"
# test_var = CollectTestData(starting_folder, ending_folder, 0.5)

# This is for a (Faster) Greedy Non-Max Suppression developed by Dr Dr. Tomasz Malisiewicz at https://tom.ai/ 
def nonMaxSuppression(boxes, overlapThresh):
    """NMS takes in the box coordinates and overlap threshold number. The assumption is that we do not have probabilities. If so we need to select those instead and change function."""
    # no boxes returns empty
    if len(boxes) == 0:
        return []
    # we want all the data to be in float format
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float") 

    # get cocordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # find the area of the box
    boxArea = (x2 - x1 + 1) *  (y2 - y1 + 1)
    index = np.argsort(x1) # we will index the list off a variable should be off prediction score though?

    # make a list for the seleted data
    selected = []
    counter = 1

    while index is not empty:
        # Take the LAST index and add it to the list of those chosen
        lastIndex = len(index) - 1 # minus one because python numbering
        pickedIndex = index[lastIndex] #the index we work with is 
        selected.append(pickedIndex)
        counter += 1
        # find the largest coordinates like with IOU BUT need maximum instead of max because maximum works across the values. 
        # Find it between the selected index and any index all the way up to the selection (which should be the end of list)
        maxX1 = np.maximum(x1[pickedIndex], x1[index[:lastIndex]])
        maxY1 = np.maximum(y1[pickedIndex], y1[index[:lastIndex]])
        minX2 = np.maximum(x2[pickedIndex], x2[index[:lastIndex]])
        minY2 = np.minimum(y2[pickedIndex], y2[index[:lastIndex]])

        # get width and height
        w = max(0.0, minX2 - maxX1 + 1) # add the 0.0 to select 0 if value is neg
        h = max(0.0, minY2 - maxY1 + 1)

        # calculate overlap ratio
        overlap = (w * h) / boxArea(index[:, lastIndex])

        # remove any indexes that are over threshold
        index = np.delete(index, np.concatenate([lastIndex],
			np.where(overlap > overlapThresh)[0]))
        # return selected bounding boxes as INTEGERS
        selected = selected[:, counter] # or is it counter -1?
    return boxes[selected].astype("int")





