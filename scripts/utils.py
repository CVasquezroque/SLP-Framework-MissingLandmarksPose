import numpy as np
import h5py
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from collections import Counter
import scipy.stats as stats
import argparse

def get_args():
    """
    Process the command line arguments.

    Returns:
    args: Namespace object that contains the parsed command line arguments
        dataset: str, the name of the dataset to process
        min_instances: int, the minimum number of instances to include a class on subsets
    """
    parser = argparse.ArgumentParser(description='Process dataset name.')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--min_instances', type=int, help='Min instances to include a class on subsets')
    parser.add_argument('--train', type=bool, default=False, help='Train Flag')
    parser.add_argument('--val', type=bool, default=False, help='Validation Flag')
    args = parser.parse_args()
    return args



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
    """Counts the number of consecutive false values in a boolean array.

    Parameters:
        arr (numpy.ndarray): The boolean array to analyze.

    Returns:
        int: The number of consecutive false values in `arr`.
    """
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
    """
    Calculates the length of the longest consecutive subsequence of `True` values in a boolean array.

    Parameters:
        arg1 (array): A boolean array.

    Returns:
        int: The length of the longest consecutive subsequence of `True` values in `arg1`.
    """
    consecutive_trues = 0
    max_consecutive_trues = 0
    for i in arg1:
        if i:
            consecutive_trues += 1
            max_consecutive_trues = max(max_consecutive_trues, consecutive_trues)
        else:
            consecutive_trues = 0
    return max_consecutive_trues
def read_h5_indexes(path):
    """
    Read data from an HDF5 file and return the classes, video names, and data as lists.
    
    Parameters:
    - path (str): a string representing the path to the HDF5 file
    
    Returns:
    - classes (list): a list of strings representing the class labels for the data
    - videoName (list): a list of strings representing the video names for the data
    - data (list): a list of arrays representing the data
    """
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


def saving_reduced_hdf5(classes, videoName, data, partial_output_name="DATASET", kp_est_chosed="mediapipe", val=False, train=False, *args, **kwargs):
    """
    Save the given classes, video names, and data to an HDF5 file, along with additional arguments if provided.
    
    Parameters:
    - classes (list): a list of strings representing the class labels for the data
    - videoName (list): a list of strings representing the video names for the data
    - data (list): a list of arrays representing the data
    - partial_output_name (str): a string to be included in the output file name (default: "DATASET")
    - kp_est_chosed (str): a string representing the chosen keypoint estimator (default: "mediapipe")
    - val (bool): a boolean indicating if the data is validation data (default: False)
    - train (bool): a boolean indicating if the data is training data (default: False)
    - args: additional arguments to be included in the HDF5 file, passed as positional arguments
    - kwargs: additional arguments to be included in the HDF5 file, passed as keyword arguments

    """
    if val and not train:
        save_path = f'../split_reduced/{partial_output_name}--{kp_est_chosed}-Val.hdf5'
    elif train and not val:
        save_path = f'../split_reduced/{partial_output_name}--{kp_est_chosed}-Train.hdf5'
    elif not val and not train:
        save_path = f'../output_reduced/{partial_output_name}--{kp_est_chosed}.hdf5'
    h5_file = h5py.File(save_path, 'w')

    for pos, (c, v, d) in enumerate(zip(classes, videoName, data)):
        grupo_name = f"{pos}"
        h5_file.create_group(grupo_name)
        h5_file[grupo_name]['video_name'] = v # video name (str)
        h5_file[grupo_name]['label'] = c # classes (str)
        h5_file[grupo_name]['data'] = d # data (Matrix)

        # Check if false_seq, percentage_groups, and max_consec were provided as arguments
        if args:
            false_seq, percentage_groups, max_consec = args
            h5_file[grupo_name]['in_range_sequences'] = false_seq # false_seq (array: int data)
            h5_file[grupo_name]['percentage_group'] = percentage_groups # percentage of reduction group (array: categorial data)
            h5_file[grupo_name]['max_percentage'] = max_consec # percentage of the max length of consecutive missing sequences (array: float data)

        # Check if false_seq, percentage_groups, and max_consec were provided as keyword arguments
        elif kwargs:
            false_seq = kwargs.get("false_seq")
            percentage_groups = kwargs.get("percentage_groups")
            max_consec = kwargs.get("max_consec")
            if false_seq:
                h5_file[grupo_name]['in_range_sequences'] = false_seq # false_seq (array: int data)
            if percentage_groups:
                h5_file[grupo_name]['percentage_group'] = percentage_groups # percentage of reduction group (array: categorial data)
            if max_consec:
                h5_file[grupo_name]['max_percentage'] = max_consec # percentage of the max length of consecutive missing sequences (array: float data)

    h5_file.close()


def plot_data(dataArr, timestep, image=None,outpath='output'):
    """
    Plot a single frame of 2D landmark data.

    Parameters:
        dataArr (np.ndarray): A 3D numpy array of landmark data of shape (timesteps, 1, num_landmarks*2)
        timestep (int): The timestep to plot
        image: An optional image to plot underneath the landmark data. Defaults to None.
        outpath (str, optional): The path to save the output plot. Defaults to 'output'.
    """
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
    """
    Parameters:
    arr_lengths (numpy.ndarray): Array of video lengths before processing.
    new_arr_lengths (numpy.ndarray): Array of video lengths after processing.
    filename (str): Name of the file to save the plot.

    Returns:
    None
    """
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
    """
    Check if any landmarks in a given slice of left or right hand are missing in any of the timesteps.

    Parameters:
    x_arr (numpy.ndarray): Array of landmark coordinates.
    left_hand_slice (slice): Slice object representing the range of the landmarks for the left hand.
    right_hand_slice (slice): Slice object representing the range of the landmarks for the right hand.

    Returns:
    all_same_landmarks (numpy.ndarray): Boolean array indicating if all landmarks in the given slice are the same
                                        in all timesteps.
    """
    left_hand_landmarks = x_arr[:,0,left_hand_slice]
    right_hand_landmarks = x_arr[:,0,right_hand_slice]
    # Check if all landmarks are the same in a timestep for left and right hand
    left_diff_arr = np.diff(left_hand_landmarks)
    right_diff_arr = np.diff(right_hand_landmarks)
    all_same_landmarks = np.all(left_diff_arr == 0, axis=1) | np.all(right_diff_arr == 0, axis=1)
    return all_same_landmarks

def getting_filtered_data(dataArrs,reduced_dataArrs, videoName, classes, min_instances = 15, banned_classes=['???', 'NNN', '']):
    """
    Filter data by selecting classes with at least min_instances instances and not in banned_classes.
    Also calculates the maximum consecutive frames where all hand landmarks are the same, the number of false
    sequences and the percentage reduction of each video after it was processed.

    Parameters:
    dataArrs (list): A list of numpy arrays representing data for each video.
    reduced_dataArrs (list): A list of numpy arrays representing reduced data for each video.
    videoName (list): A list of strings representing the names of each video.
    classes (list): A list of strings representing the class labels for each video.
    min_instances (int): Minimum number of instances for each class to be considered valid. Default is 15.
    banned_classes (list): A list of strings representing the class labels to be banned from the filtered data. Default is ['???', 'NNN', ''].

    """
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
    """
    Filter data by selecting classes with at least min_instances instances and not in banned_classes.

    Parameters:
    dataArrs (list): A list of numpy arrays representing data for each video.
    reduced_dataArrs (list): A list of numpy arrays representing reduced data for each video.
    videoName (list): A list of strings representing the names of each video.
    classes (list): A list of strings representing the class labels for each video.
    min_instances (int): Minimum number of instances for each class to be considered valid. Default is 15.
    banned_classes (list): A list of strings representing the class labels to be banned from the filtered data. Default is ['???', 'NNN', ''].

    """
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
    """
    Filters the data in an HDF5 file based on whether the frames have the same landmarks on the left and right hands.

    Args:
        h5_path (str): Path to the HDF5 file.
        left_hand_slice (slice): Slice of landmarks to use for the left hand. Default is `slice(31, 50)`.
        right_hand_slice (slice): Slice of landmarks to use for the right hand. Default is `slice(51,70)`.
        consecutive_trues (callable): Function that determines whether there are consecutive frames with the same landmarks. 
            Should take a 1D numpy boolean array as input and return a boolean. Default is `None`.
        filtering_videos (bool): Whether to filter out videos where all frames have missing landmarks. Default is `True`.

    Returns:
        - The filtered classes.
        - The filtered video names.
        - The filtered data arrays with missing landmarks removed.
        - The original data arrays with missing landmarks preserved (but without videos with zero frames after reducing missing landmarks).
        
    """
    classes, videoName, dataArrs = read_h5_indexes(h5_path)
    print("Size of data:",len(dataArrs))
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
