import numpy as np
import argparse
from utils import read_h5_indexes, filter_same_landmarks, filter_data

#########
# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Process dataset name.')
# Add an argument for dataset name
parser.add_argument('--dataset', type=str, help='Dataset name')
parser.add_argument('--min_instances', type=int, help='Min instances to include a class on subsets')
# Parse the command-line arguments
args = parser.parse_args()

########

DATASET = args.dataset
KPMODEL = 'mediapipe'
print(DATASET)
print(KPMODEL)

h5_path = f'../output/{DATASET}--{KPMODEL}.hdf5'

classes, videoName, dataArrs = read_h5_indexes(h5_path)

arrClasses = np.array(classes)
arrVideoNames = np.array(videoName, dtype=object)
arrData = np.array(dataArrs, dtype=object)

print(arrData.shape)

has_zeros = lambda arg: np.any(np.sum(arg, axis=0) == 0) #For a time step has zeros or not
has_consecutive_trues = lambda arg1,arg2: np.any(np.convolve(arg1.astype(int), np.ones(arg2), mode='valid') >= arg2)



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
#PUCP
# self.list_labels_banned = ["ya", "qué?", "qué", "bien", "dos", "ahí", "luego", "yo", "él", "tú","???","NNN"]
# self.list_labels_banned += ["sí","ella","uno","ese","ah","dijo","llamar"]
new_classes, new_videoName, new_arrData,arrData_without_empty,max_consec = filter_same_landmarks(h5_path,left_hand_slice=slice(501, 521), right_hand_slice=slice(522,542))
print(f"Mean value of max consecutive frames with missing landmarks {np.mean(max_consec['Max']):.2f} for {DATASET} dataset")
print(f"Mean value of percentage of consecutive frames with missing landmarks {np.mean(max_consec['Max Percentage']):.2f} for {DATASET} dataset")

fdataArrs, fvideoNames, fclasses, fvalid_classes, fvalid_classes_total = filter_data(arrData, videoName, classes, min_instances = min_instances, banned_classes=bann)
filtered_dataArrs, filtered_videoNames, filtered_classes, valid_classes,valid_classes_total  = filter_data(arrData_without_empty, new_videoName, new_classes, min_instances = min_instances, banned_classes=bann)
_, _, fnew_classes, fnew_valid_classes, fnew_valid_classes_total = filter_data(arrData_without_empty, new_videoName, new_classes, min_instances = min_instances, banned_classes=[])
filtered_dataArrs2, filtered_videoNames2, filtered_classes2, valid_classes2, valid_classes_total2 = filter_data(new_arrData, new_videoName, new_classes, min_instances = min_instances, banned_classes=bann)


########################
reduceArray = lambda arg: np.sum(arg, axis=1) # sums elements in the array along given axis (axis=1 symbolizes sum over columns).
sumFlags = lambda arg: np.sum(arg) # sums all elements in the array.

evaluateMissing= lambda arg: np.where(arg==0,1,0) # evaluates if any element in the array equals 0, if yes then replaces it with 1, if no then keep it 0.
evaluateOutRange = lambda arg: np.where((arg<0) | (1<arg),1,0) # evaluates whether any element in the array is <0 or >1, if yes then replaces it with 1, if no then keeps it 0.

lOutRange = list(map(evaluateOutRange, dataArrs[0:2])) # maps evaluateOutrange function over each item of 'dataArs' list.
lFinalOutRange = list(map(sumFlags,lOutRange)) # maps sumFlags functions over each item of 'lOutRange' list.
arrFinalOutRange = np.array(lFinalOutRange) # converts list to array. Have the items where have out of range points


missing_landmarks = lambda arr: np.invert(np.all(np.diff(arr, axis=1) == 0, axis=2))
lMissing = list(map(missing_landmarks,arrData[0:2]))
lFinalMissing = list(map(sumFlags,lMissing)) # maps sumFlags functions over each item of 'lOutRange' list.
arrFinalMissing = np.array(lFinalMissing) # converts list to array. Have the items where have out of range points

print(lMissing)
print(arrFinalMissing)


print(lOutRange)
print(lOutRange[0].shape)
print(arrFinalOutRange)