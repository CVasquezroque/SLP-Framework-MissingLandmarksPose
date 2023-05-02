import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import read_h5_indexes, filter_same_landmarks, filter_data, saving_reduced_hdf5, getting_filtered_data

DATASET = 'AEC'
KPMODEL = 'mediapipe'
print(DATASET)
print(KPMODEL)

h5_path = f'../output_reduced/{DATASET}--{KPMODEL}-Val.hdf5'

classes, videoName, dataArrs = read_h5_indexes(h5_path)

arrClasses = np.array(classes)
arrVideoNames = np.array(videoName, dtype=object)
arrData = np.array(dataArrs, dtype=object)

print(arrData.shape)

has_zeros = lambda arg: np.any(np.sum(arg, axis=0) == 0) #For a time step has zeros or not
has_consecutive_trues = lambda arg1,arg2: np.any(np.convolve(arg1.astype(int), np.ones(arg2), mode='valid') >= arg2)

min_instances = 15
bann = ["ya", "qué?", "qué", "bien", "dos", "ahí", "luego", "yo", "él", "tú","???","NNN"]
if DATASET == "AEC":
    bann = ["ya", "qué?", "qué", "bien", "dos", "ahí", "luego", "yo", "él", "tú","???","NNN"]
elif DATASET == "AUTSL":
    bann = ['apple','computer','fish','kiss','later','no','orange','pizza','purple','secretary','shirt','sunday','take','water','yellow']
elif DATASET == "PUCP_PSL_DGI156":
    bann = ["ya", "qué?", "qué", "bien", "dos", "ahí", "luego", "yo", "él", "tú","???","NNN"]
    bann += ["sí","ella","uno","ese","ah","dijo","llamar"]
else:
    bann = ["ya", "qué?", "qué", "bien", "dos", "ahí", "luego", "yo", "él", "tú","???","NNN"]

new_classes, new_videoName, new_arrData,arrData_without_empty = filter_same_landmarks(h5_path,left_hand_slice=slice(501, 521), right_hand_slice=slice(522,542))
# print(f"Mean value of max consecutive frames with missing landmarks {np.mean(max_consec['Max']):.2f} for {DATASET} dataset")
# print(f"Mean value of percentage of consecutive frames with missing landmarks {np.mean(max_consec['Max Percentage']):.2f} for {DATASET} dataset")

fdataArrs, fvideoNames, fclasses, fvalid_classes, fvalid_classes_total = filter_data(arrData, videoName, classes, min_instances = min_instances, banned_classes=bann)
filtered_dataArrs, filtered_videoNames, filtered_classes, valid_classes,valid_classes_total = filter_data(arrData_without_empty, new_videoName, new_classes, min_instances = min_instances, banned_classes=bann)
_, _, fnew_classes, fnew_valid_classes, fnew_valid_classes_total = filter_data(arrData_without_empty, new_videoName, new_classes, min_instances = min_instances, banned_classes=[])
filtered_dataArrs2, filtered_videoNames2, filtered_classes2, valid_classes2, valid_classes_total2 = filter_data(new_arrData, new_videoName, new_classes, min_instances = min_instances, banned_classes=bann)

print("#################################")
print("arrData Original Dataset with all videos")
print('classes:',len(fclasses))
print('valid_classes:',len(fvalid_classes))
print('valid_classes with non min instances:',len(fvalid_classes_total))
print("#################################")
print("Filtered same landmarks, baseline subset (0 frames videos substracted) without banning")
print('classes:',len(fnew_classes))
print('valid_classes:',len(fnew_valid_classes))
print('valid_classes with non min instances:',len(fnew_valid_classes_total))
print("#################################")
print("Filtered same landmarks, baseline subset (0 frames videos substracted) with banning")
print('classes:',len(filtered_classes))
print('valid_classes:',len(valid_classes))
print('valid_classes with non min instances:',len(valid_classes_total))
print("#################################")
print("Filtered same landmarks, reduced subset with banning")
print('classes:',len(filtered_classes2))
print('valid_classes:',len(valid_classes2))
print('valid_classes with non min instances:',len(valid_classes_total2))

filtered_dataArrs,filtered_reduceArrs, filtered_videoNames, filtered_classes, valid_classes, valid_classes_total, max_consec,num_false_seq,percentage_reduction_categories = getting_filtered_data(arrData_without_empty,new_arrData,new_videoName,new_classes,min_instances= min_instances,banned_classes=bann)

for i in [filtered_dataArrs,filtered_reduceArrs, filtered_videoNames, filtered_classes, valid_classes, valid_classes_total, max_consec,num_false_seq,percentage_reduction_categories]:
    print(f'{len(i)}')

#Saving Baseline subset
saving_reduced_hdf5(
    filtered_classes,
    filtered_videoNames,
    num_false_seq,
    percentage_reduction_categories,
    max_consec,
    filtered_dataArrs,
    partial_output_name=f"{DATASET}--added_features")
#Saving Reduced subset
saving_reduced_hdf5(
    filtered_classes,
    filtered_videoNames,
    num_false_seq,
    percentage_reduction_categories,
    max_consec,
    filtered_dataArrs,
    partial_output_name=f"{DATASET}_reduced--added_features")
