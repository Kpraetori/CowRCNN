# import libraries
import os
import random
import cv2
import pandas as pd
import numpy as np

from helperfunctions import calculate_IOU, visualiseSS, save_ROI, cowfaceBB, BBconversion, collect_test_data

# Sets overall data directory path
DIR_PATH = "sourceData" # should rename to source data

# Dictates the MAXIMUM number or region proposals allowed for training and inference (at the end)
REGION_PROPOSALS = 2000 # test number use 2000 in final version
FINAL_PROPOSALS = 200

# Dictates how many images should be created out of each original image
# Want a positive bias per Girshick
MAX_COWFACE_YES = 30
MAX_COWFACE_NO = 3

# Input dimensions based on MobileNet v2 requirements
INPUT_DIMS = (224, 224)

# set selective segmentation
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# Total counts for cow faces and counts for background images. Used for naming convention
countCowFaces = 0
countBackground = 0
imagecount = 0

# Where the new files will be saved
cowPath = 'results/CowFaces'
bkgPath = 'results/Background'

# Convert YOLO to X1,Y1,X2,Y2 and get the image
cowBBandImage = BBconversion(DIR_PATH)

for groundTruthBB, image in cowBBandImage:
    print(f'ground truth labels: {groundTruthBB}')
    # testBB = cowfaceBB(image, groundTruthBB) # prints image with bb

    # selective search over the image
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    boxes = ss.process()

    # proposed Regions list
    proposedRegions = []
    for x,y,w,h in boxes:
        # print(x,y,w,h)
        proposedRegions.append((x, y, x + w, y + h))

    # testSS = visualiseSS(image, boxes)
    # make a counter that connects to cowface yes and no counts. Do ROI & IOU up to those numbers
    yesCowROI = 0
    noCowROI = 0

    for boxRegion in proposedRegions[:REGION_PROPOSALS]: 
        for BB in groundTruthBB:
            iou = calculate_IOU(BB, boxRegion)
            # print(f'IOU is {iou}')
            
            # while noCowROI < MAX_COWFACE_NO and yesCowROI < MAX_COWFACE_YES:
            # IOU > 0.5 was used by Girshick et al.
            if iou > 0.5 and yesCowROI < MAX_COWFACE_YES:
                print(f"Face detected at IOU {iou}")
                (regionStartX, regionStartY, regionEndX, regionEndY) = boxRegion
                print(f"The ground truth is {BB}")
                print(f"The box region from selective search is {boxRegion}")
                roi = image[regionStartY:regionEndY, regionStartX:regionEndX]
                save_ROI(roi, countCowFaces, cowPath, Face=True)
                countCowFaces += 1
                yesCowROI += 1
    
            elif iou < 0.3 and noCowROI < MAX_COWFACE_NO:
                print("Background detected")
                (regionStartX, regionStartY, regionEndX, regionEndY) = boxRegion
                roi = image[regionStartY:regionEndY, regionStartX:regionEndX]
                save_ROI(roi, countBackground, bkgPath, Face=False)
                countBackground += 1
                noCowROI += 1
        if noCowROI == (MAX_COWFACE_NO - 1) and yesCowROI == (MAX_COWFACE_YES -1):
            break
    print(f'Yes Cow ROI count is {yesCowROI}')
    print(f'No Cow ROI count is {noCowROI}')
    print(f'Total background images {countBackground}')
    print(f'Total cow face images {countCowFaces}')
    imagecount += 1
    print(f'Finished with loop. {imagecount} processed so far.')

    # break              


# get_background_testdata = collect_test_data()
# get_cowface_testdata = collect_test_data()