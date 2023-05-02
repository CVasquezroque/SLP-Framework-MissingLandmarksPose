import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from utils import *
import cv2
import matplotlib.animation as animation
import matplotlib.widgets as widgets

def read_h5_indexes(path,indexesLandmarks):

    classes = []
    videoName = []
    data = []

    #read file
    with h5py.File(path, "r") as f:
        for index in f.keys():
            classes.append(f[index]['label'][...].item().decode('utf-8'))
            videoName.append(f[index]['video_name'][...].item().decode('utf-8'))
            data.append(f[index]["data"][:,:,indexesLandmarks-1])
    
    return classes, videoName, data

def plot_right_hand_displacement(x, y):
    # Select 10 random sample points from the right hand
    sample_points = np.random.choice(range(502, 522), size=20, replace=False)

    # Compute the Euclidean distance between each sample point and the previous frame
    # displacements = np.sqrt((x[1:, 0, sample_points] - x[:-1, 0, sample_points])**2 +
    #                         (y[1:, 0, sample_points] - y[:-1, 0, sample_points])**2)


    # Plot the displacement of each sample point against the frame number
    colors = ['red', 'blue', 'green', 'purple']
    for i, p in enumerate(sample_points):
        plt.subplot(4,5,i+1)
        plt.plot(x[:,0,p],y[:,0,p], label=f'Point {p}', alpha=0.5, color=colors[(i-1)%4])
        plt.title(f'Point {p}')
        plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

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
#PUCP_PSL_DGI305
#PUCP_PSL_DGI156
PATH = 'datasets\\AEC\\Videos\\SEGMENTED_SIGN\\'
OUT_PATH = 'datasets\\AEC\\Videos\\VISUALIZATION\\'
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)
file_path = "./output/AEC--mediapipe.hdf5"

points = pd.read_csv('D:\\Home\\ConnectingPoints\\points_71.csv')
origin = points.origin
target = points.tarjet
bodyPart = points.tar_name

def get_mp_keys(points):
    tar = np.array(points.mp_pos)-1
    return list(tar)

poseModel = 'mediapipe'
indexesLandmarks = np.array(points.mp_pos)


classes, videoName, data = read_h5_indexes(file_path,indexesLandmarks)

df = create_df(file_path)

print(classes[0],videoName[0],data[0].shape)

glosses, videoName, data = read_h5_indexes(file_path,indexesLandmarks)

df = pd.DataFrame.from_dict({
    "classes":glosses,
    "videoName": videoName,
    "data":data,  
})

###############################

colors = ['red', 'blue', 'green', 'purple']
labels = ['Whole Pose', 'Face', 'Left Hand', 'Right Hand']


# input_file = "ATARDECER_ORACION_2\\IR_4.mp4"
input_file = "proteinas_porcentajes\\cuánto_1287.mp4"
set_frames = []
for index, row in df.iterrows():
    n_frames = row['data'].shape[0]
    set_frames.append(n_frames)
set_frames = np.array(set_frames)
print(f'Min: {np.min(set_frames)}, Max: {np.max(set_frames)}, Mean: {np.mean(set_frames)}, Std: {np.std(set_frames)}, p25: {np.percentile(set_frames,25)}, p75: {np.percentile(set_frames,75)}')
for index, row in df.iterrows():
    # filename = PATH + row['videoName']
    filename = PATH + row['videoName']

    # filename = PATH + 'Historias_vinetas_14\\'+ row['videoName']
    if row['videoName'].startswith('comer'):
        print(filename)
    if row['videoName'] == input_file:
        video_frames = obtain_frames(filename)
        sample_data = row['data']
        x, y = np.split(sample_data, 2, axis=1)
        # Create a VideoWriter object
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)
        im = ax.imshow(video_frames[0, :, :, :])
        def update(i):
            im.set_data(video_frames[i, :, :, :])
            ax.clear()
            ax.imshow(video_frames[i, :, :, :])
            for j, (start, end) in enumerate([(0,8),(9,30),(31,50),(51,70)]): #Should be 0,32 but 26-32 are legs (0, 25), (33, 501), (502,522),(523,543)
                if labels[j] == 'Right Hand' or  labels[j] == 'Left Hand':
                    print(f'Frame #{i}')
                    print('x:',np.unique(x[i,0,start:end],return_counts=True))
                    print('y:',np.unique(y[i,0,start:end],return_counts=True))
                    print(np.unique(x[i,0,start:end]).shape[0])
                    print(np.unique(y[i,0,start:end]).shape[0])
                ax.scatter(x[i,0,start:end]*video_frames[0,:,:,:].shape[1], y[i,0,start:end]*video_frames[0,:,:,:].shape[0], s=10, c=colors[j], alpha=0.5)
                for k, (x_val, y_val) in enumerate(zip(x[i,0,start:end], y[i,0,start:end])):
                    if labels[j] == 'LeftHand':
                        ax.annotate(f"{k + start + 1}", (x_val * video_frames[0,:,:,:].shape[1], y_val * video_frames[0,:,:,:].shape[0]), color=colors[j])
            ax.set_title(f'Frame #{i+1}')
            return im

        axcolor = 'lightgoldenrodyellow'
        axframe = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        sframe = widgets.Slider(axframe, 'Frame', 1, video_frames.shape[0], valinit=1, valstep=1)

        def update_frame(val):
            update(int(val))
            fig.canvas.draw_idle()

        sframe.on_changed(update_frame)
        plt.show()
        # plt.figure(2,figsize=(10,10))
        # plot_right_hand_displacement(x,y)



###########################
# Generating the videos
# colors = ['red', 'blue', 'green', 'purple']
# labels = ['Whole Pose', 'Face', 'Left Hand', 'Right Hand']
# for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='Processing Videos', bar_format='{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}'):
#     filename = PATH + row['videoName']
#     video_frames = obtain_frames(filename)
#     sample_data = row['data']
#     x, y = np.split(sample_data, 2, axis=1)
#     if not os.path.exists(OUT_PATH + row['videoName'].split('\\')[0]):
#         os.makedirs(OUT_PATH + row['videoName'].split('\\')[0])
#     if not os.path.exists(OUT_PATH + row['videoName']):
#         # Create a VideoWriter object
#         fig = plt.figure()
#         writer = animation.FFMpegWriter(fps=10, bitrate=5000)
#         with writer.saving(fig,OUT_PATH + row['videoName'], dpi=300):
#             for i in tqdm(range(video_frames.shape[0]),desc=row['videoName'], total=video_frames.shape[0], bar_format='{l_bar}{bar}[{elapsed}<{remaining},{rate_fmt}]',leave=False):
#                 plt.clf()
#                 plt.imshow(video_frames[i,:,:,:])
#                 for j, (start, end) in enumerate([(0, 25), (33, 501), (502,522),(523,543)]): #Should be 0,32 but 26-32 are legs
#                     plt.scatter(x[i,0,start:end]*video_frames[0,:,:,:].shape[1], y[i,0,start:end]*video_frames[0,:,:,:].shape[0], s=10, c=colors[j], alpha=0.5)
#                     for k, (x_val, y_val) in enumerate(zip(x[i,0,start:end], y[i,0,start:end])):
#                         if labels[j] == 'Whole Pose':
#                             plt.annotate(f"{k + start + 1}", (x_val * video_frames[0,:,:,:].shape[1], y_val * video_frames[0,:,:,:].shape[0]), color=colors[j])
#                 writer.grab_frame()
#             plt.close()







# for index, row in df.iterrows():
#     filename = PATH + row['videoName']
#     if row['videoName'] == input_file:
#         video_frames = obtain_frames(filename)
#         sample_data = row['data']
#         x, y = np.split(sample_data, 2, axis=1)
#         # Create a VideoWriter object
#         fig = plt.figure()
#         def update(i):
#             plt.clf()
#             plt.imshow(video_frames[i,:,:,:])
#             for j, (start, end) in enumerate([(0, 25), (33, 501), (502,522),(523,543)]): #Should be 0,32 but 26-32 are legs
#                 plt.scatter(x[i,0,start:end]*video_frames[0,:,:,:].shape[1], y[i,0,start:end]*video_frames[0,:,:,:].shape[0], s=10, c=colors[j], alpha=0.5)
#                 for k, (x_val, y_val) in enumerate(zip(x[i,0,start:end], y[i,0,start:end])):
#                     if labels[j] == 'Whole Pose':
#                         plt.annotate(f"{k + start + 1}", (x_val * video_frames[0,:,:,:].shape[1], y_val * video_frames[0,:,:,:].shape[0]), color=colors[j])
#             plt.title(f'Frame #{i}')
#         ani = animation.FuncAnimation(fig,update,frames = video_frames.shape[0],repeat=False)
#         plt.show()



