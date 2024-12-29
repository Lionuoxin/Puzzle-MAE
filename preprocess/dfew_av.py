# *_*coding:utf-8 *_*
import os
import pandas as pd
import sys
import glob
import torchaudio
import numpy as np


data_path = "/datasets/Dynamic-expression/DFEW"
split_dir = os.path.join(data_path, 'EmoLabel_DataSplit')
video_dir = os.path.join(data_path, '/datasets/Dynamic-expression/DFEW/Clip/clip_224x224')
audio_dir = os.path.join(data_path, '/space/lixin/DFEW/Clip/audio_16k')
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

# label, 11 single-labeled emotions
labels = ["happy", "sad", "neutral", "angry", "surprise", "disgustfear", "fear"]
label2idx = {idx:l for idx, l in enumerate(labels)}

# read split file
for split in splits:
    save_dir = f'./saved/data/dfew/audio_visual/split0{split}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_split_file = os.path.join(split_dir, f'train(single-labeled)/set_{split}.csv')
    df = pd.read_csv(train_split_file, header=0, delimiter=',')
    train_label_dict = dict(zip(df.iloc[:, 0][1:], df.iloc[:, 1][1:]))

    test_split_file = os.path.join(split_dir, f'test(single-labeled)/set_{split}.csv')
    df = pd.read_csv(test_split_file, header=None, delimiter=',')
    test_label_dict = dict(zip(df.iloc[:, 0][1:], df.iloc[:, 1][1:]))

    train_label_list, test_label_list = [], []
    for v, l in train_label_dict.items(): # ex:1600 3
        part = (int(v)-1)//1600 + 1
        sample_name = os.path.join(f"clip_224x224_part_{part}", "{:0>5}".format(v))
        video_file = os.path.join(video_dir, sample_name)
        audio_file = os.path.join(audio_dir, f"part_{part}/{v}.{audio_file_ext}")
        if not os.path.exists(audio_file):
            print(f"Warning: the audio of sample '{sample_name}' in split {split} training set does not exist, pass it!")
            continue
        train_label_list.append([video_file, audio_file, int(l)-1])
    for v, l in test_label_dict.items(): # ex:1600 3
        part = (int(v)-1)//1600 + 1
        sample_name = os.path.join(f"clip_224x224_part_{part}", "{:0>5}".format(v))
        video_file = os.path.join(video_dir, sample_name)
        audio_file = os.path.join(audio_dir, f"part_{part}/{v}.{audio_file_ext}")
        if not os.path.exists(audio_file):
            print(f"Warning: the audio of sample '{sample_name}' in split {split} test set does not exist, pass it!")
            continue
        test_label_list.append([video_file, audio_file, int(l)-1])

    total_samples = len(train_label_list) + len(test_label_list)
    print(f'Total samples in split {split}: {total_samples}, train={len(train_label_list)}, test={len(test_label_list)}')

    # write
    new_train_split_file = os.path.join(save_dir, f'train.csv')
    df = pd.DataFrame(train_label_list)
    df.to_csv(new_train_split_file, header=None, index=False, sep=' ')

    new_test_split_file = os.path.join(save_dir, f'test.csv')
    df = pd.DataFrame(test_label_list)
    df.to_csv(new_test_split_file, header=None, index=False, sep=' ')

