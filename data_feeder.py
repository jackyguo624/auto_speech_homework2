import numpy as np
import librosa
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


wav_dir_prefix = "./data/ASVspoof2017_"
sample_rate = 16000
n_fft = int(25 * sample_rate / 1000)
hop_length = int(10 * sample_rate / 1000)


def load_label(label_file):
    labels = {}
    encode = {'genuine': 0, 'spoof': 1}
    with open(label_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) > 1:
                wav_id = line[0].replace(".wav", "")
                tmp_label = encode[line[1]]
                labels[wav_id] = tmp_label
    return labels


def load_data(dataset, label_file):
    labels = load_label(label_file)
    dataset_dir = wav_dir_prefix + dataset

    final_data = []
    final_label = []
    final_wav_ids = []

    for root, _, file_names in os.walk(dataset_dir):
        for file_name in tqdm(file_names, desc="loading {} data".format(dataset)):
            wav_path = os.path.join(root, file_name)
            wav_id = wav_path.split("/")[-1].split('.')[0]
            label = labels[wav_id]
            audio, _ = librosa.load(wav_path, sr=sample_rate,)
            feature = librosa.feature.mfcc(audio, sr=sample_rate, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
            final_data.append(feature)
            final_label.append(label)
            final_wav_ids.append(wav_id)
    return final_data, final_label, final_wav_ids


class ASVDataSet(Dataset):

    def __init__(self, data, label, wav_ids, transform=True):
        super(ASVDataSet, self).__init__()
        self.data = data
        self.label = label
        self.wav_ids = wav_ids
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        each_data, each_label, each_wav_id = self.data[idx], self.label[idx], self.wav_ids[idx]
        if self.transform:
            each_data, each_label = torch.from_numpy(each_data), torch.LongTensor([each_label])
        return {
            "data": each_data,
            "label": each_label,
            "wav_id": each_wav_id
        }








