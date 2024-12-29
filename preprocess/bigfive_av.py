# *_*coding:utf-8 *_*
import os
import pandas as pd
import sys
import glob
import torchaudio
import numpy as np


data_path = "/space0/lixin/project/big_five/big_five"
video_dir = os.path.join(data_path, 'frames1')
audio_dir = os.path.join(data_path, 'voice_data')
audio_sample_rate = 16000 # expected
audio_file_ext = 'wav'

num_splits = 5
splits = range(1, num_splits + 1)

# check video
# total_samples = 10045
video_dirs = sorted(glob.glob(os.path.join(video_dir, '*')))
# assert len(video_dirs) == total_samples, f'Error: wrong number of videos, expected {total_samples}, got {len(video_dirs)}.'

# check audio and its sample rate
audio_files = glob.glob(os.path.join(audio_dir, '*.wav'))
# total_audios = 10024 # ffmpeg can not extract audios from 21 samples
# assert len(audio_files) == total_audios, f'Error: wrong number of audios, expected {total_audios}, got {len(audio_files)}.'
# audio_durations = []
# for audio_file in audio_files:
#     wav, sr = torchaudio.load(audio_file)
#     assert sr == audio_sample_rate, f"Error: '{audio_file}' has a sample rate of {sr}, expected {audio_sample_rate}!"
#     audio_durations.append(wav.shape[1] / audio_sample_rate)
# print(f'Audio duration: mean={np.mean(audio_durations):.1f}s, max={max(audio_durations):.1f}s, min={min(audio_durations):.1f}s.')

# read split file
save_dir = f'./saved/data/bigfuve/audio_visual/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
train_file = os.path.join(data_path, 'dataset/y_label_training.csv')
df = pd.read_csv(train_file, header=0, delimiter=',')
train_label_dict = dict(zip(list(df.iloc[:, 0]), np.array(df.iloc[:, 1:])))

test_file = os.path.join(data_path, 'dataset/y_label_test.csv')
df = pd.read_csv(test_file, header=None, delimiter=',')
test_label_dict = dict(zip(list(df.iloc[:, 0]), np.array(df.iloc[:, 1:])))

train_label_list, test_label_list = [], []

for v, l in train_label_dict.items(): # --Ymqszjv54.000,(0.4205607476635514,0.5625,0.6593406593406592,0.47572815533980584,0.5777777777777777,0.4579439252336448)
    sample_name = v[0:-4]
    video_file = os.path.join(video_dir, "train_frames", sample_name)
    audio_file = os.path.join(audio_dir, "train", f"{sample_name}.{audio_file_ext}")
    if not os.path.exists(audio_file):
        print(f"Warning: the audio of sample '{sample_name}' in training set does not exist, pass it!")
        continue
    train_label_list.append([video_file, audio_file, l[0], l[1], l[2], l[3], l[4]])
    
for v, l in test_label_dict.items(): # ex:1600 3
    sample_name = v[0:-4]
    video_file = os.path.join(video_dir, "test_frames", sample_name)
    audio_file = os.path.join(audio_dir, "test", f"{sample_name}.{audio_file_ext}")
    if not os.path.exists(audio_file):
        print(f"Warning: the audio of sample '{sample_name}' in test set does not exist, pass it!")
        continue
    test_label_list.append([video_file, audio_file, l[0], l[1], l[2], l[3], l[4]])
    
total_samples = len(train_label_list) + len(test_label_list)


print(f'Total samples : {total_samples}, train={len(train_label_list)}, test={len(test_label_list)}')
# write
new_train_split_file = os.path.join(save_dir, f'train.csv')
df = pd.DataFrame(train_label_list)
df.to_csv(new_train_split_file, header=None, index=False, sep=' ')
new_test_split_file = os.path.join(save_dir, f'test.csv')
df = pd.DataFrame(test_label_list)
df.to_csv(new_test_split_file, header=None, index=False, sep=' ')
