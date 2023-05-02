import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from utils import read_h5_indexes, show_statistics


points = pd.read_csv('/home/cc/repositories/ConnectingPoints/points_71.csv')
poseModel = 'mediapipe'
indexesLandmarks = np.array(points.mp_pos)

h5_path = '/home/cc/repositories/ConnectingPoints/split/AEC--mediapipe-Train.hdf5'

classes, videoName, dataArrs = read_h5_indexes(h5_path,indexesLandmarks)


arrClasses = np.array(classes)
arrVideoNames = np.array(videoName, dtype=object)
arrData = np.array(dataArrs, dtype=object)
ik = 0
print(f"Samples:{arrClasses[ik]} {arrVideoNames[ik]} {arrData[ik].shape}")


database = 'AEC'
frame_path = f'/home/cc/repositories/ConnectingPoints/datasets/{database}/Videos/SEGMENTED_SIGN/{arrVideoNames[ik]}'
print(frame_path)
# video_frames = obtain_frames(frame_path)
# plot_data(arrData[ik],0,video_frames[ik],'output_AEC_0')
#utilities to summarize landmarks of an instance
reduceArray = lambda arg: np.sum(arg, axis=1) # sums elements in the array along given axis (axis=1 symbolizes sum over columns).
sumFlags = lambda arg: np.sum(arg) # sums all elements in the array.

evaluateMissing= lambda arg: np.where(arg==0,1,0) # evaluates if any element in the array equals 0, if yes then replaces it with 1, if no then keep it 0.
evaluateOutRange = lambda arg: np.where((arg<0) | (1<arg),1,0) # evaluates whether any element in the array is <0 or >1, if yes then replaces it with 1, if no then keeps it 0.

lOutRange = list(map(evaluateOutRange, dataArrs)) # maps evaluateOutrange function over each item of 'dataArs' list.
# arrOutRange = np.array(lOutRange) # converts list to array.
lFinalOutRange = list(map(sumFlags,lOutRange)) # maps sumFlags functions over each item of 'lOutRange' list.
arrFinalOutRange = np.array(lFinalOutRange) # converts list to array. Have the items where have out of range points

arrAuxMissing = map(reduceArray,dataArrs) # sum the coordinates x and y to evaluate if the sum is still 0, which helps evaluate the landmark as missing
arrMissing = map(evaluateMissing,arrAuxMissing) # evaluate 1s in the 2D array (reduced in the dimensionality for the coordinates) and sum them to count the number of missing landmarks (0,0)
lFinalMissing = list(map(sumFlags,arrMissing))
arrFinalMissing = np.array(lFinalMissing)

#total_missing_landmarks = np.sum(arrFinalMissing)
countCoordinates = lambda arg: np.prod(np.array(arg.shape))/2 # to divide by third dimension of two coordinates per landmark
lCountLandmarks = list(map(countCoordinates,dataArrs))
arrCountLandmarks = np.array(lCountLandmarks)


arrPercentageMissing = arrFinalMissing / arrCountLandmarks
arrPercentageOutRange = arrFinalOutRange / arrCountLandmarks

arrMisOrOutRange = arrFinalMissing + arrFinalOutRange
arrMisNotOutRange_bool = np.logical_and(arrFinalMissing>0,np.logical_not(arrFinalOutRange>0))
arrMisNotOutRange = np.where(arrMisNotOutRange_bool, arrFinalMissing, 0)

arrPercentageMissOrOut = arrMisOrOutRange/arrCountLandmarks
arrPercentageMissNotOut = arrMisNotOutRange/arrCountLandmarks

print(arrPercentageMissNotOut*100)
print("Statistics of MissNotOut Videos")
show_statistics(arrPercentageMissNotOut*100)
plt.figure()
plt.hist(arrPercentageMissNotOut, bins=50, label='Percentage of MissNotOutRange Videos')
plt.legend(loc='upper right')
plt.xlabel('Video length')
plt.ylabel('Count')
plt.savefig('Histogram')

nonzero_indices = np.nonzero(arrPercentageMissNotOut)
arr_nonzero = arrPercentageMissNotOut[nonzero_indices]

# Plot the histogram
plt.figure()
plt.hist(arr_nonzero, bins=50, label='Percentage of MissNotOutRange Videos')
plt.legend(loc='upper right')
plt.xlabel('Video length')
plt.ylabel('Count')
plt.savefig('Histogram_nonzero')

has_zeros = lambda arg: np.any(np.sum(arg, axis=0) == 0) #For a time step has zeros or not
has_consecutive_trues = lambda arg1,arg2: np.any(np.convolve(arg1.astype(int), np.ones(arg2), mode='valid') >= arg2)


def check_consecutive_missing(dataArrs,num_consecutive):
    not_consecutive_missing_landmarks = []
    has_zeros = lambda x: np.any(np.sum(x, axis=0) == 0)
    has_consecutive_trues = lambda x: np.any(np.convolve(x.astype(int), np.ones(num_consecutive), mode='valid') >= num_consecutive)

    for i, arr in enumerate(dataArrs):
        timesteps_with_zeros = np.array([has_zeros(x) for x in arr])
        consecutive_trues = np.array([has_consecutive_trues(timesteps_with_zeros[j:j+num_consecutive]) for j in range(len(timesteps_with_zeros)-(num_consecutive-1))])
        if not np.any(consecutive_trues):
            not_consecutive_missing_landmarks.append(i)
        
    return np.array(not_consecutive_missing_landmarks)

not_consecutive_missing_landmarks = check_consecutive_missing(dataArrs,3)

indices = np.where(arrMisNotOutRange)[0]
arrData = np.array(dataArrs, dtype=object)

# print(indices.shape
# print(np.intersect1d(not_consecutive_missing_landmarks,indices))
# print(np.intersect1d(not_consecutive_missing_landmarks,indices).shape)
non_consecutive_missing_landmark_index = np.intersect1d(not_consecutive_missing_landmarks,indices)
arrData_non_consecutive = arrData[non_consecutive_missing_landmark_index]


# create a copy of arrData to preserve the original data
arrData_copy = arrData.copy()

# remove frames with missing landmarks for videos with indices in non_consecutive_missing_landmark_index
compute_length = lambda arg: arg.shape[0]

for i in non_consecutive_missing_landmark_index:
    # get the indices of frames with missing landmarks for this video
    missing_frames = np.unique(np.where(np.sum(arrData[i], axis=1) == 0)[0])
    # remove the frames with missing landmarks
    if missing_frames.shape[0]>0:
        arrData_copy[i] = np.delete(arrData_copy[i], missing_frames, axis=0)
    
length_videos_arrData = list(map(compute_length,arrData))
length_videos_arrData_without_missing = list(map(compute_length,arrData_copy))

# print(length_videos_arrData[:10])
# print(length_videos_arrData_without_missing[:10])
# print(non_consecutive_missing_landmark_index[:10])

# print(length_videos_arrData[-10:])
# print(length_videos_arrData_without_missing[-10:])
# print(non_consecutive_missing_landmark_index[-10:])

plt.figure()
plt.hist(length_videos_arrData, bins=50, alpha=0.5, label='With missing frames')
plt.hist(length_videos_arrData_without_missing, bins=50, alpha=0.5, label='Without missing frames')
plt.legend(loc='upper right')
plt.xlabel('Video length')
plt.ylabel('Count')
plt.savefig('Histogram_length')


print("Statistics for length of Videos from arrData")
show_statistics(length_videos_arrData)
print("Statistics for length of Videos from arrData without Missing")
show_statistics(length_videos_arrData_without_missing)

percent_reduction = (sum(length_videos_arrData) - sum(length_videos_arrData_without_missing)) / (sum(length_videos_arrData)) * 100
print(f'Percentage of Reduction of the DataSet: {percent_reduction}%')


zero_indices = [i for i, length in enumerate(length_videos_arrData_without_missing) if length == 0]

if len(zero_indices) > 0:
    print("There are {} videos with no frames after removing missing landmarks".format(len(zero_indices)))
else:
    print("All videos have frames after removing missing landmarks")

zero_length_videos = [i for i, length in enumerate(length_videos_arrData_without_missing) if length == 0]
for i in zero_length_videos:
    print(f"Video {videoName[i]} has length {length_videos_arrData[i]} in the original data.")