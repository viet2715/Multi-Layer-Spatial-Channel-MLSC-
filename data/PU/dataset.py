from scipy.io import loadmat
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

def segment_signal(signal, window_length=4096, overlap_ratio=0.25):
    step = int(window_length * (1 - overlap_ratio))
    segments = []
    for start in range(0, len(signal) - window_length + 1 , step):
        segments.append(signal[start:start + window_length])
    return segments
all_segments = []


for i in range(1, 21): # 20 files per class
    path = f'PU/KB27/N15_M07_F04_KB27_{i}.mat'  # File path to the .mat file
    data = loadmat(path,struct_as_record=False, squeeze_me=True)
    link = 'N15_M07_F04_KB27_' + f'{i}'
    signal = data[link].Y[6].Data
    segments = segment_signal(signal)
    print(len(segments))
    all_segments.append(segments)

    print()
all_segments = np.vstack(all_segments)
