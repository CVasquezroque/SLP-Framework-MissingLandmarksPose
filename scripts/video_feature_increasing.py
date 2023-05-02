import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import read_h5_indexes, filter_same_landmarks, saving_reduced_hdf5, getting_filtered_data, get_args

args = get_args()

DATASET = args.dataset
KPMODEL = 'mediapipe'

VAL = args.val
TRAIN = args.train
print(DATASET)
print(KPMODEL)
print(f'Validation Flag set to {VAL} and Train Flag set to {TRAIN}')

if VAL and not TRAIN:
    h5_path = f'../split_reduced/{DATASET}--{KPMODEL}-Val.hdf5'
elif TRAIN and not VAL:
    h5_path = f'../split_reduced/{DATASET}--{KPMODEL}-Train.hdf5'
else:
    h5_path = f'../output_reduced/{DATASET}--{KPMODEL}.hdf5'

classes, videoName, dataArrs = read_h5_indexes(h5_path)

arrClasses = np.array(classes)
arrVideoNames = np.array(videoName, dtype=object)
arrData = np.array(dataArrs, dtype=object)


has_zeros = lambda arg: np.any(np.sum(arg, axis=0) == 0) #For a time step has zeros or not
has_consecutive_trues = lambda arg1,arg2: np.any(np.convolve(arg1.astype(int), np.ones(arg2), mode='valid') >= arg2)

if not VAL and not TRAIN:
    min_instances = args.min_instances
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
else:
    min_instances = 0
    bann = []

new_classes, new_videoName, new_arrData,arrData_without_empty = filter_same_landmarks(h5_path,left_hand_slice=slice(501, 521), right_hand_slice=slice(522,542))

filtered_dataArrs,filtered_reduceArrs, filtered_videoNames, filtered_classes, valid_classes, valid_classes_total, max_consec,num_false_seq,percentage_reduction_categories = getting_filtered_data(arrData_without_empty,new_arrData,new_videoName,new_classes,min_instances= min_instances,banned_classes=bann)


print("*"*20)
print("Checking all sets have the same size")
for i in [filtered_dataArrs,filtered_reduceArrs, filtered_videoNames, filtered_classes, max_consec,num_false_seq,percentage_reduction_categories]:
    print(f'{len(i)}')

#Saving Baseline subset
saving_reduced_hdf5(
    filtered_classes,
    filtered_videoNames,
    filtered_dataArrs,
    partial_output_name=f"{DATASET}--added_features",
    val=VAL,
    train=TRAIN,
    false_seq = num_false_seq,
    percentage_group = percentage_reduction_categories,
    max_consec = max_consec)
#Saving Reduced subset
saving_reduced_hdf5(
    filtered_classes,
    filtered_videoNames,
    filtered_dataArrs,
    partial_output_name=f"{DATASET}_reduced--added_features",
    val=VAL,
    train=TRAIN,
    false_seq = num_false_seq,
    percentage_group = percentage_reduction_categories,
    max_consec = max_consec)
