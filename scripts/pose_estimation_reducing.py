import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from utils import read_h5_indexes, filter_same_landmarks, filter_data, saving_reduced_hdf5, plot_length_distribution, get_args

args = get_args()

DATASET = args.dataset
KPMODEL = 'mediapipe'
print(DATASET)
print(KPMODEL)

h5_path = f'../output/{DATASET}--{KPMODEL}.hdf5'
if not os.path.exist('../output_reduced'):
    os.makedirs('../output_reduced')
if not os.path.exist('../split_reduced'):
    os.makedirs('../split_reduced')

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
new_classes, new_videoName, new_arrData,arrData_without_empty = filter_same_landmarks(h5_path,left_hand_slice=slice(501, 521), right_hand_slice=slice(522,542))

fdataArrs, fvideoNames, fclasses, fvalid_classes, fvalid_classes_total= filter_data(arrData, videoName, classes, min_instances = min_instances, banned_classes=bann)
filtered_dataArrs, filtered_videoNames, filtered_classes, valid_classes,valid_classes_total = filter_data(arrData_without_empty, new_videoName, new_classes, min_instances = min_instances, banned_classes=bann)
_, _, fnew_classes, fnew_valid_classes, fnew_valid_classes_total= filter_data(arrData_without_empty, new_videoName, new_classes, min_instances = min_instances, banned_classes=[])
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



#Saving Baseline subset
saving_reduced_hdf5(filtered_classes,filtered_videoNames,filtered_dataArrs,partial_output_name=DATASET)
#Saving Reduced subset
saving_reduced_hdf5(filtered_classes2,filtered_videoNames2,filtered_dataArrs2,partial_output_name=f"{DATASET}_reduced")

####################
# Assuming original dataset lists: classes, videoName, arrData
# Assuming filtered dataset lists: new_classes, new_videoName, new_arrData


# # Step 1: Extract relevant information for each video
# video_info = []

# for i in range(len(new_videoName)):
#     original_video_length = arrData_without_empty[i].shape[0]
#     if new_videoName[i] in new_videoName:
#         index_new_arr = new_videoName.index(new_videoName[i])
#         filtered_video_length = new_arrData[index_new_arr].shape[0]

#         # Calculate number of missing landmark frames
#         num_missing_landmark_frames = original_video_length - filtered_video_length

#         # Calculate percentage of missing landmark frames
#         perc_missing_landmark_frames = (num_missing_landmark_frames / original_video_length) * 100

#     else:
#         num_missing_landmark_frames = original_video_length
#         perc_missing_landmark_frames = 100
    
#     video_info.append({
#         'name': new_videoName[i],
#         'class': new_classes[i],
#         'original_length': original_video_length,
#         'missing_landmark_frames': num_missing_landmark_frames,
#         'perc_missing_landmark_frames': perc_missing_landmark_frames
#     })

# # Step 2: Aggregate data and analyze the distribution of missing landmarks

# # Extract the percentage of missing landmark frames for all videos
# perc_missing_landmark_frames_all = [video['perc_missing_landmark_frames'] for video in video_info]

# # Calculate mean and median percentage of missing landmark frames
# mean_perc_missing_landmark_frames = np.mean(perc_missing_landmark_frames_all)
# median_perc_missing_landmark_frames = np.median(perc_missing_landmark_frames_all)

# print(f"Mean Percentage of Missing Landmark Frames: {mean_perc_missing_landmark_frames}")
# print(f"Median Percentage of Missing Landmark Frames: {median_perc_missing_landmark_frames}")
# sns.set(font_scale=1.2, font='serif')
# sns.set_style('whitegrid')
# # Plot a histogram of the percentage of missing landmark frames
# sns.histplot(perc_missing_landmark_frames_all, stat='percent', bins=10, kde=False, edgecolor='black')
# plt.xlabel('Percentage of Frames with Missing Landmarks')
# plt.ylabel(f'% of Total Frames in Dataset {DATASET} ')
# plt.savefig(f"ESANN_2023/Figures/{DATASET}_Histogram_Distribution.png",dpi=300)


# # # Define the total width and height of each rectangle
# # total_width = 100
# # rect_height = 9

# # # Define the color for non-missing landmarks
# # non_missing_color = '#e6e6e6'

# # # Calculate the width of each rectangle
# # widths = [perc/100 * total_width for perc in perc_missing_landmark_frames_all]

# # # Plot the rectangles
# # fig, ax = plt.subplots(figsize=(10, 7))
# # rects1 = ax.barh(y=range(100, 0, -10), width=total_width, height=rect_height, color=non_missing_color, edgecolor='black')
# # rects2 = ax.barh(y=range(100, 0, -10), width=widths, height=rect_height, color='red', edgecolor='black')

# # # Customize the plot
# # ax.set_xlim(0, 100)
# # ax.set_ylim(0, 105)
# # ax.set_yticks(range(0, 110, 10))
# # ax.set_yticklabels([f'{i}%' for i in range(0, 110, 10)])
# # ax.set_xlabel('Percentage of Frames with Missing Landmarks')
# # ax.set_ylabel(f'% of Total Frames in Dataset {DATASET} ')
# # ax.set_title('Histogram of Percentage of Missing Landmark Frames')
# # plt.show()


# arr_lengths = np.array(list(map(lambda x: x.shape[0], arrData)))
# new_arr_lengths = np.array(list(map(lambda x: x.shape[0], new_arrData)))
# plot_length_distribution(arr_lengths,new_arr_lengths,f'ESANN_2023/Figures/{DATASET}_length_distribution_v1.png')

# arr_lengths = np.array(list(map(lambda x: x.shape[0], filtered_dataArrs)))
# new_arr_lengths = np.array(list(map(lambda x: x.shape[0], filtered_dataArrs2)))

# plot_length_distribution(arr_lengths,new_arr_lengths,f'ESANN_2023/Figures/{DATASET}_length_distribution_v2.png')



# # First, define the filtered_classes
# # Then, create an empty dictionary
# plt.figure()
# class_dict = {}

# # Loop through each class in filtered_classes
# for n,c in enumerate(filtered_classes):
#     # Initialize empty lists for original and reduced lengths
#     class_dict[c] = {
#             'Original Length': [],
#             'Reduced Length': []
#         }
# print(class_dict.keys())
# print(len(filtered_dataArrs))
# print(len(filtered_dataArrs2))
# for n,arr in enumerate(filtered_dataArrs):
#     c = filtered_classes[n]
#     class_dict[c]['Original Length'].append(arr.shape[0])
#     class_dict[c]['Reduced Length'].append(filtered_dataArrs2[n].shape[0])
# print(class_dict)


# df = []
# for c in class_dict:
#     for l in ['Original Length', 'Reduced Length']:
#         for v in class_dict[c][l]:
#             df.append({'Class': c, 'Length': l, 'Value': v})
# df = pd.DataFrame(df)


# print(df.head())
# # Create a new column with the start letter of each class name
# try:
#     df['Start Letter'] = df['Class'].str[0]
# except:
#     df['Start Letter'] = df['Class']
# # Sort the DataFrame by the 'Start Letter' column and the 'Value' column
# df = df.sort_values(['Start Letter', 'Value'])


# print(df.head())



# # Set font size and family
# sns.set(font_scale=1.75, font='Times New Roman')
# sns.set_style('whitegrid')

# # Create nested boxplot
# sns.boxplot(data=df, x='Value', y='Class', hue='Length')
# plt.xlabel('Length')
# plt.ylabel('Class')
# # plt.show()
# plt.savefig(f'ESANN_2023/Figures/{DATASET}_boxplot.png')



# # Calculate the percentage reduction
# df['Percentage Reduction'] = (1 - (df['Value'] / df.groupby('Class')['Value'].transform('mean'))) * 100

# # Define the percentage reduction groups
# bins = [0, 20, 40, 60, 100]
# labels = ['0-20%', '20-40%', '40-60%', '80-100%']

# # Group the instances into percentage reduction groups
# df['Percentage Reduction Group'] = pd.cut(df['Percentage Reduction'], bins=bins, labels=labels)


# plt.figure(figsize=(15,10))
# # Set the Seaborn style
# sns.set_style('whitegrid')

# # Create a violin plot with the percentage reduction group as x, original length as y, and gray palette
# sns.violinplot(x='Percentage Reduction Group', y='Value', data=df, palette='gray')

# # Set the x label and title

# plt.ylabel('\% of Frames Reduction')
# plt.title(f'Impact of Reduction of Frames with Missing Landmarks in {DATASET}')


# # Show the plot
# plt.savefig(f'ESANN_2023/Figures/{DATASET}_violin_plot.png')



# plt.figure(figsize=(15,10))
# # Set font size and family
# sns.set(font_scale=1.75, font='Times New Roman')
# sns.set_style('whitegrid')

# # Create a new column in the dataframe for percentage reduction group
# df['Percentage Reduction Group'] = pd.cut(df['Percentage Reduction'], bins=[0, 20, 40, 60, 100],
#                                           labels=['0-20%', '20-40%', '40-60%', '80-100%'])

# # Create nested boxplot
# sns.boxplot(data=df, y='Value', x='Percentage Reduction Group', hue='Length')
# plt.ylabel('Length')
# plt.xlabel('Percentage Reduction Group')
# plt.title(f'Impact of reduction of frames with missing landmarks in {DATASET}')
# plt.legend(title='Length')
# plt.savefig(f'ESANN_2023/Figures/{DATASET}_nested_boxplot.png')

#########################


    #     class_label = data_orig[0][0][0]
    #     # If the class label matches the current filtered class, add the length to the appropriate list
    #     if class_label == c:
    #         orig_lengths.append(data_orig.shape[0])
    #         reduced_lengths.append(data_reduced.shape[0])
    # # Add the lists to the class_dict dictionary with the current filtered class as the key
    # class_dict[c] = [orig_lengths, reduced_lengths]

# Create a swarm plot


# Create a pandas dataframe from the class_dict
# Calculate the mean value of video length for each class
# df['Original Mean'] = df['Original Length'].apply(lambda x: sum(x)/len(x))
# df['Reduced Mean'] = df['Reduced Length'].apply(lambda x: sum(x)/len(x))
# df['Original SD'] = df['Original Length'].apply(lambda x: np.std(x))
# df['Reduced SD'] = df['Reduced Length'].apply(lambda x: np.std(x))
# df['Original Max'] = df['Original Length'].apply(lambda x: max(x))
# df['Reduced Max'] = df['Reduced Length'].apply(lambda x: max(x))
# df['Original Min'] = df['Original Length'].apply(lambda x: min(x))
# df['Reduced Min'] = df['Reduced Length'].apply(lambda x: min(x))

# # Calculate means and standard deviations for original and reduced lengths
# # Add mean and standard deviation columns for each length type
# df['Mean'] = df.groupby(['Class', 'Length'])['Value'].transform('mean')
# df['SD'] = df.groupby(['Class', 'Length'])['Value'].transform('std')

# # Set seaborn style and font
# sns.set_style('whitegrid')
# sns.set(font_scale=1.25, font='Times New Roman')

# # Create a nested plot using catplot
# g = sns.catplot(
#     data=df,
#     x='Mean',
#     y='Class',
#     hue='Length',
#     kind='point',
#     join=False,
#     dodge=True,
#     capsize=0.2,
#     height=10,
#     aspect=0.8
# )

# # # Add error bars for standard deviation
# # for i, row in enumerate(g.ax.get_children()):
# #     if i % 2 == 1:
# #         err = df.iloc[(i-1)//2]['SD']
# #         x = row.get_xdata()
# #         y = row.get_ydata()
# #         g.ax.errorbar(x, y, xerr=err, fmt='none', color='black')

# # Set plot labels
# plt.xlabel('Video Length')
# plt.ylabel('Class')

# # Show the plot
# plt.show()



# Step 1: Extract relevant information for each video
# video_info = []

# for i in range(len(videoName)):
#     original_video_length = arrData[i].shape[0]
#     if videoName[i] in new_videoName:
#         index_new_arr = new_videoName.index(videoName[i])
#         filtered_video_length = new_arrData[index_new_arr].shape[0]

#         # Calculate number of missing landmark frames
#         num_missing_landmark_frames = original_video_length - filtered_video_length

#         # Calculate percentage of missing landmark frames
#         perc_missing_landmark_frames = (num_missing_landmark_frames / original_video_length) * 100

#     else:
#         num_missing_landmark_frames = original_video_length
#         perc_missing_landmark_frames = 100
    
#     video_info.append({
#         'name': videoName[i],
#         'class': classes[i],
#         'original_length': original_video_length,
#         'missing_landmark_frames': num_missing_landmark_frames,
#         'perc_missing_landmark_frames': perc_missing_landmark_frames
#     })
# print(video_info)

# # Step 2: Aggregate data and analyze the distribution of missing landmarks

# # Extract the percentage of missing landmark frames for all videos
# perc_missing_landmark_frames_all = [video['perc_missing_landmark_frames'] for video in video_info]

# # Calculate mean and median percentage of missing landmark frames
# mean_perc_missing_landmark_frames = np.mean(perc_missing_landmark_frames_all)
# median_perc_missing_landmark_frames = np.median(perc_missing_landmark_frames_all)

# print(f"Mean Percentage of Missing Landmark Frames: {mean_perc_missing_landmark_frames}")
# print(f"Median Percentage of Missing Landmark Frames: {median_perc_missing_landmark_frames}")

# # Plot a histogram of the percentage of missing landmark frames
# plt.hist(perc_missing_landmark_frames_all, bins=10, edgecolor='black')
# plt.xlabel('Percentage of Missing Landmark Frames')
# plt.ylabel('Number of Videos')
# plt.title('Histogram of Percentage of Missing Landmark Frames')
# plt.savefig('percentage_missing_landmark_frames_histogram.png')
# plt.show()

# #########################
# # Assuming the video_info list created in Step 1


# sns.set_style('whitegrid')
# sns.set(font_scale=1.25, font='Times New Roman')

# # Step 3: Analyze video length variation in the total dataset
# original_lengths = [video['original_length'] for video in video_info]
# filtered_lengths = [video['original_length'] - video['missing_landmark_frames']
#                     for video in video_info]

# # Calculate mean and median video length before and after filtering
# mean_original_length = np.mean(original_lengths)
# median_original_length = np.median(original_lengths)
# mean_filtered_length = np.mean(filtered_lengths)
# median_filtered_length = np.median(filtered_lengths)

# print(f"Mean Original Video Length: {mean_original_length}")
# print(f"Median Original Video Length: {median_original_length}")
# print(f"Mean Filtered Video Length: {mean_filtered_length}")
# print(f"Median Filtered Video Length: {median_filtered_length}")

# # Plot box plots comparing video length distributions before and after filtering
# data = {'Original': original_lengths, 'Filtered': filtered_lengths}
# # Convert data to a long-form DataFrame
# data = pd.DataFrame(data)

# print(data)
# print(len(video_info))
# print(video_info[0].keys())
# print(data.keys())
# print(len(data))

# fig, ax = plt.subplots(figsize=(8, 6))

# sns.catplot(data=data, kind="box")
# ax.set_ylabel('Video Length', fontsize=14, fontname='Times New Roman')
# ax.set_title('Box Plot of Video Lengths Before and After Filtering', fontsize=16, fontname='Times New Roman')

# fig.savefig('video_lengths_boxplot.png')
# plt.show()

# # Step 4: Video length variation and missing landmark distribution per class

# from collections import defaultdict

# class_video_info = defaultdict(list)
# selected_classes = []
# i = 0
# for video in video_info:
#     print("video from video_info:",video)
#     class_video_info[video['class']].append(video)
#     i+=1
#     if i>2:
#         break
# print(class_video_info)
# for cls, videos in class_video_info.items():
#     if len(videos) >= 15:
#         selected_classes.append(cls)
#         original_lengths_cls = [video['original_length'] for video in videos]
#         filtered_lengths_cls = [video['original_length'] - video['missing_landmark_frames']
#                                 for video in videos]

#         # Plot box plots comparing video length distributions before and after filtering for the class
#         data_cls = {'Original': original_lengths_cls, 'Filtered': filtered_lengths_cls}
#         data_cls = pd.DataFrame(data_cls)
#         fig_cls, ax_cls = plt.subplots(figsize=(8, 6))
        
#         sns.boxplot(data=data_cls, width=0.5)
#         ax_cls.set_ylabel('Video Length', fontsize=14, fontname='Times New Roman')
#         ax_cls.set_title(f"Box Plot of Video Lengths for Class {cls} Before and After Filtering", fontsize=16, fontname='Times New Roman')

#         fig_cls.savefig(f'video_lengths_class_{cls}_boxplot.png')
#         plt.show()

# selected_classes now contains the classes that have at least 15 instances for analysis
