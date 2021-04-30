import os, shutil
import numpy as np
import random
import glob

# Rename files from cowYOLO raw data

def rename_files(folderPath):
    """This function removes duplicate file names by renaming image files based on the name of the parent folder (ending characters of the folder name)"""
    files = os.listdir(folderPath)
    for file in files:
        folderPathEnd = folderPath[-6:]
        print(f"Folder path ending is {folderPathEnd}")
        filePrefix = os.path.splitext(file)[0]
        fileExt = os.path.splitext(file)[1]
        newName = folderPathEnd + filePrefix
        os.rename(os.path.join(folderPath, file), os.path.join(folderPath, (newName + fileExt)))
        print(f"File {filePrefix} is renamed to {newName}")
        

# data = "cowdata/file48YOLO"
# test = rename_files(data)

# Mini version of source data
dir_path = "sourceData"

def collect_data_subset(filePath, destinationPath, percent: float):
    """This function is to take a percent of the dataset and move it to a new location to split the dataset. The images chosen are from a randomised list."""
        files = os.listdir(filePath)
        # need list for shuffling later
        name_list = []
        # go through all files
        for file_name in files:
            # first look for the txt files so we don't pull duplicates
            if file_name.endswith('.txt'):
                filePrefix = os.path.splitext(file_name)[0]
                # make a list to sort through later
                name_list.append(filePrefix)
        # Once we have all the names, shuffle the list
        random.shuffle(name_list)
        # find the number needed
        selection = int(len(name_list)*percent)
        # make a list containing the number needed
        selected_list = name_list[:selection]
        # go through the list and pull the files
        for file in files:
            fileName = os.path.splitext(file)[0]
            if fileName in selected_list:
                shutil.move(os.path.join(filePath, file), destinationPath)
                print(f"File {fileName} is moved")
            else:
                print(f"File {fileName} remains")
  
# starting_folder = "sourceData"
# ending_folder = "sourceDataSubset"
# move = collect_data_subset(starting_folder, ending_folder, 0.1)

