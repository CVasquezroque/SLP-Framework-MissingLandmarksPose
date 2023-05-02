import numpy as np
import h5py
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from collections import Counter
import scipy.stats as stats

def obtain_frames(filename):
    cap = cv2.VideoCapture(filename)
    video_frames = []
    if (cap.isOpened() == False):
        print('Error')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frames.append(frame)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    return np.array(video_frames)

def count_false_sequences(arr):
    count = 0
    curr_false_count = 0
    for i in range(len(arr)):
        if not arr[i]:
            curr_false_count += 1
        else:
            if curr_false_count > 0:
                count += 1
            curr_false_count = 0
    if curr_false_count > 0:
        count += 1
    return count
def max_consecutive_trues(arg1):
    consecutive_trues = 0
    max_consecutive_trues = 0
    for i in arg1:
        if i:
            consecutive_trues += 1
            max_consecutive_trues = max(max_consecutive_trues, consecutive_trues)
        else:
            consecutive_trues = 0
    return max_consecutive_trues

def saving_reduced_hdf5(classes,videoName,false_seq,percentage_groups,max_consec,data,partial_output_name = "DATASET",kp_est_chosed = "mediapipe",val=False,train=False):
    if val and not train:
        save_path = f'../split_reduced/{partial_output_name}--{kp_est_chosed}-Val.hdf5'
    elif train and not val:
        save_path = f'../split_reduced/{partial_output_name}--{kp_est_chosed}-Train.hdf5'
    elif not val and not train:
        save_path = f'../output_reduced/{partial_output_name}--{kp_est_chosed}.hdf5'
    h5_file = h5py.File(save_path, 'w')

    for pos, (c, v, d, f, p, m) in enumerate(zip(classes, videoName, data,false_seq,percentage_groups,max_consec)):
        grupo_name = f"{pos}"
        h5_file.create_group(grupo_name)
        h5_file[grupo_name]['video_name'] = v # video name (str)
        h5_file[grupo_name]['label'] = c # classes (str)
        h5_file[grupo_name]['data'] = d # data (Matrix)
        h5_file[grupo_name]['in_range_sequences'] = f # false_seq (array: int data)
        h5_file[grupo_name]['percentage_group'] = p # percentage of reduction group (array: categorial data)
        h5_file[grupo_name]['max_percentage'] = m # percentage of the max length of consecutive missing sequences (array: float data)
        
    h5_file.close()


def obtain_frames(filename):
    cap = cv2.VideoCapture(filename)
    video_frames = []
    if (cap.isOpened() == False):
        print('Error')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frames.append(frame)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    return np.array(video_frames)

def read_h5_indexes(path):

    classes = []
    videoName = []
    data = []
    try:
        #read file
        with h5py.File(path, "r") as f:
            for index in f.keys():
                classes.append(f[index]['label'][...].item().decode('utf-8'))
                videoName.append(f[index]['video_name'][...].item().decode('utf-8'))
                data.append(f[index]["data"][...]) #indexesLandmarks
    except:
            #read file
        with h5py.File(path, "r") as f:
            for index in f.keys():
                classes.append(f[index]['label'][...].item())
                videoName.append(f[index]['video_name'][...].item().decode('utf-8'))
                data.append(f[index]["data"][...]) #indexesLandmarks    
    return classes, videoName, data

def plot_data(dataArr, timestep, image=None,outpath='output'):
    x, y = np.split(dataArr, 2, axis=1)
    plt.figure()
    if image is not None:
        resized_img = cv2.resize(image, (2, 2), interpolation=cv2.INTER_AREA)
        plt.imshow(resized_img)
        size = 1
    else:
        plt.imshow(np.zeros((2,2)))   
        plt.xlim([0,1])
        plt.ylim([0,1])
        size = 1
    plt.scatter(x[timestep,0,:]*size, y[timestep,0,:]*size, s=10, alpha=0.5, c='b')
    # Add annotations to each landmark
    for i in range(len(x[0,0,:])):
        plt.annotate(str(i), (x[timestep,0,i]*size, y[timestep,0,i]*size), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.savefig(outpath)

def plot_length_distribution(arr_lengths  , new_arr_lengths, filename):
    # plot histograms using seaborn
    sns.set_style('whitegrid')
    sns.set(font_scale=1.2, font='serif')
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.histplot(arr_lengths,bins=50,ax=ax, kde=False,label='Before', color='#c1272d', alpha=0.8)
    sns.histplot(new_arr_lengths,bins=50,ax=ax,kde=False, label='After', color='#0000a7', alpha=0.5)

    ax.set_xlabel('Video length (number of frames)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of video lengths')
    ax.legend()
    plt.savefig(filename,dpi=300)
    plt.show() 

def show_statistics(array):
    mean = np.mean(array)
    median = np.median(array)
    percentile_25 = np.percentile(array, 25)
    percentile_75 = np.percentile(array, 75)
    min_value = np.min(array)
    max_value = np.max(array)

    print('Mean:', mean)
    print('Median:', median)
    print('25th Percentile:', percentile_25)
    print('75th Percentile:', percentile_75)
    print('Minimum:', min_value)
    print('Maximum:', max_value)

def get_missing_landmarks(x_arr,left_hand_slice,right_hand_slice):
    left_hand_landmarks = x_arr[:,0,left_hand_slice]
    right_hand_landmarks = x_arr[:,0,right_hand_slice]
    # Check if all landmarks are the same in a timestep for left and right hand
    left_diff_arr = np.diff(left_hand_landmarks)
    right_diff_arr = np.diff(right_hand_landmarks)
    all_same_landmarks = np.all(left_diff_arr == 0, axis=1) | np.all(right_diff_arr == 0, axis=1)
    return all_same_landmarks

def getting_filtered_data(dataArrs,reduced_dataArrs, videoName, classes, min_instances = 15, banned_classes=['???', 'NNN', '']):
    # Get class counts
    class_counts = Counter(classes)
    
    # Select classes with >= min_instances instances and not in banned_classes
    valid_classes = [cls for cls, count in class_counts.items() if count >= min_instances and cls not in banned_classes]
    valid_classes_total = [cls for cls, count in class_counts.items() if cls not in banned_classes]
    # Filter dataArrs and videoNames based on valid_classes
    filtered_dataArrs = []
    filtered_reduceArrs = []
    filtered_videoNames = []
    filtered_classes = []
    max_consec = {'Max': [], 'Max Percentage': []}
    num_false_seq = []

    for i, cls in enumerate(classes):
        if cls in valid_classes:
            
            x_arr, y_arr = np.split(dataArrs[i], 2, axis=1)
            all_same_landmarks = get_missing_landmarks(x_arr, slice(501, 521), slice(522,542))
            max_consec_value = max_consecutive_trues(all_same_landmarks)
            num_false_seq.append(count_false_sequences(all_same_landmarks))
            max_consec['Max'].append(max_consec_value) if max_consec_value != 0 else None
            max_consec['Max Percentage'].append(max_consec_value / len(all_same_landmarks))

            
            filtered_dataArrs.append(dataArrs[i])
            filtered_reduceArrs.append(reduced_dataArrs[i])
            filtered_videoNames.append(videoName[i])
            filtered_classes.append(cls)
    
    # Calculate percentage reduction for each video
    baseline_lengths = np.array(list(map(lambda x: x.shape[0], filtered_dataArrs)))
    reduced_lengths = np.array(list(map(lambda x: x.shape[0], filtered_reduceArrs)))
    percentage_reductions = ((baseline_lengths - reduced_lengths) / baseline_lengths) * 100
    # labels = ['0-20%', '20-40%', '40-60%','60-80%','80-100%']
    bins = [0, 20, 40, 60, 80, 100]
    # import pandas as pd
    # Categorize percentage reductions and return them in an array
    # percentage_reduction_categories = pd.cut(percentage_reductions, bins=bins, labels=labels, right=False)
    percentage_reduction_categories = np.digitize(percentage_reductions, bins=bins) - 1

    return filtered_dataArrs,filtered_reduceArrs, filtered_videoNames, filtered_classes, valid_classes, valid_classes_total, max_consec['Max Percentage'],num_false_seq,percentage_reduction_categories


def filter_data(dataArrs, videoName, classes, min_instances = 20, banned_classes=['???', 'NNN', '']):
    # Get class counts
    class_counts = Counter(classes)
    
    # Select classes with >= min_instances instances and not in banned_classes
    valid_classes = [cls for cls, count in class_counts.items() if count >= min_instances and cls not in banned_classes]
    valid_classes_total = [cls for cls, count in class_counts.items() if cls not in banned_classes]
    # Filter dataArrs and videoNames based on valid_classes
    filtered_dataArrs = []
    filtered_videoNames = []
    filtered_classes = []
    for i, cls in enumerate(classes):
        if cls in valid_classes:
            filtered_dataArrs.append(dataArrs[i])
            filtered_videoNames.append(videoName[i])
            filtered_classes.append(cls)
    
    return filtered_dataArrs, filtered_videoNames, filtered_classes, valid_classes, valid_classes_total


def filter_same_landmarks(h5_path, left_hand_slice=slice(31, 50), right_hand_slice=slice(51,70), consecutive_trues=None, filtering_videos=True):
    
    classes, videoName, dataArrs = read_h5_indexes(h5_path)
    print(len(dataArrs))
    arrData = np.array(dataArrs, dtype=object)

    new_arrData = []
    arrData_without_empty = []
    new_classes = []
    new_videoName = []
    max_consec = {}
    max_consec['Max'] = []
    max_consec['Max Percentage'] = []
    num_false_seq = []
    for n, arr in enumerate(arrData):
        x_arr, y_arr = np.split(arr, 2, axis=1)
        all_same_landmarks = get_missing_landmarks(x_arr,left_hand_slice,right_hand_slice)
        
        num_consec = (len(all_same_landmarks)//4 )*2 +1
        max_consec_value = max_consecutive_trues(all_same_landmarks)
        num_false_seq.append(count_false_sequences(all_same_landmarks))
        max_consec['Max'].append(max_consec_value) if max_consec_value!=0 else None
        max_consec['Max Percentage'].append(max_consec_value/len(all_same_landmarks)) if max_consec_value!=0 else None


        if consecutive_trues is not None:
            has_consecutive_same_landmarks = consecutive_trues(all_same_landmarks, num_consec)
        else:
            has_consecutive_same_landmarks = False

        if has_consecutive_same_landmarks:
            all_same_landmarks = np.full((len(all_same_landmarks),), True)

        
        if filtering_videos:
            if not np.all(all_same_landmarks): #Si todos los frames tienen missing landmarks no pasa
                arrData_without_empty.append(arr)
                new_classes.append(classes[n])
                new_videoName.append(videoName[n])
                
                mask = np.invert(all_same_landmarks)

                filtered_x_arr = x_arr[mask]
                filtered_y_arr = y_arr[mask]
                filtered_arr = np.concatenate((filtered_x_arr, filtered_y_arr), axis=1)
                new_arrData.append(filtered_arr)
        else:
            arrData_without_empty.append(arr)
            new_classes.append(classes[n])
            new_videoName.append(videoName[n])
            mask = np.invert(all_same_landmarks)

            filtered_x_arr = x_arr[mask]
            filtered_y_arr = y_arr[mask]
            filtered_arr = np.concatenate((filtered_x_arr, filtered_y_arr), axis=1)
            new_arrData.append(filtered_arr)
    return new_classes,new_videoName,np.array(new_arrData,dtype=object),np.array(arrData_without_empty,dtype=object) #Modificar las estadisticas para que cuadre con esto
