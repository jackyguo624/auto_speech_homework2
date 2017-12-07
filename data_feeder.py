import numpy as np
import librosa
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


wav_dir_prefix = "./data/ASVspoof2017_"
sample_rate = 16000
extend_num = 5
n_fft = int(25 * sample_rate / 1000)
hop_length = int(10 * sample_rate / 1000)


def load_label(label_file):
    labels = {}
    wav_lists = []
    encode = {'genuine': 0, 'spoof': 1}
    with open(label_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) > 1:
                wav_lists.append(line[0])
                wav_id = line[0].replace(".wav", "")
                tmp_label = encode[line[1]]
                labels[wav_id] = tmp_label
    return labels, wav_lists


def load_data(dataset, label_file, mode="train"):
    labels, wav_lists = load_label(label_file)
    dataset_dir = wav_dir_prefix + dataset

    if mode == "train":
        final_data = []
        final_label = []

        for wav_name in tqdm(wav_lists, desc="{} data".format(dataset)):
            wav_path = os.path.join(dataset_dir, wav_name)
            wav_id = wav_path.split("/")[-1].split('.')[0]
            label = labels[wav_id]
            audio, _ = librosa.load(wav_path, sr=sample_rate,)
            # get 39 dim feature
            mfcc = librosa.feature.mfcc(audio, sr=sample_rate, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
            mfcc_delta_1 = librosa.feature.delta(mfcc)
            mfcc_delta_2 = librosa.feature.delta(mfcc_delta_1)
            feature = np.concatenate((mfcc, mfcc_delta_1, mfcc_delta_2), axis=0)
            # do frame extend
            feature = np.pad(feature, [[0, 0], [extend_num-1, extend_num-1]], mode="edge")
            for i in range(4, feature.shape[1] - 5):
                tmp_feature = feature[:, i-4:i+5].reshape(-1)
                final_data.append(tmp_feature)
                final_label.append(label)
        return final_data, final_label

    elif mode == "test":
        final_data = []
        final_label = []
        final_wav_ids = []

        for wav_name in tqdm(wav_lists, desc="{} data".format(dataset)):
            wav_path = os.path.join(dataset_dir, wav_name)
            wav_id = wav_path.split("/")[-1].split('.')[0]
            label = labels[wav_id]
            audio, _ = librosa.load(wav_path, sr=sample_rate,)
            mfcc = librosa.feature.mfcc(audio, sr=sample_rate, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
            mfcc_delta_1 = librosa.feature.delta(mfcc)
            mfcc_delta_2 = librosa.feature.delta(mfcc_delta_1)
            feature = np.concatenate((mfcc, mfcc_delta_1, mfcc_delta_2), axis=0)
            # do frame extend
            feature = np.pad(feature, [[0, 0], [extend_num - 1, extend_num - 1]], mode="edge")
            final_feature = []
            for i in range(4, feature.shape[1] - 5):
                tmp_feature = feature[:, i-4:i+5].reshape(-1)
                final_feature.append(tmp_feature)
            final_feature = np.array(final_feature).astype(np.float32)
            final_data.append(final_feature)
            final_label.append(label)
            final_wav_ids.append(wav_id)

        return final_data, final_label, final_wav_ids

    else:
        raise ValueError("the mode doesn't exist")


class ASVDataSet(Dataset):

    def __init__(self, data, label, wav_ids=None, transform=True, mode="train"):
        super(ASVDataSet, self).__init__()
        self.data = data
        self.label = label
        self.wav_ids = wav_ids
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode == "train":
            each_data, each_label = self.data[idx], self.label[idx]
        else:
            each_data, each_label, each_wav_id = self.data[idx], self.label[idx], self.wav_ids[idx]
        if self.transform:
            each_data, each_label = torch.from_numpy(each_data).float(), torch.LongTensor([each_label])
        return {
            "data": each_data,
            "label": each_label
        } if self.mode == "train" else {
            "data": each_data,
            "label": each_label,
            "wav_id": each_wav_id
        }








