import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import torchaudio
import logging
import pandas as pd
import os

class CsvAudioDataset(Dataset):
    def __init__(self, csv_path, audio_root_dir, target_sr, target_sr_se, audio_col='filepath_deg', label_col='mos'):
        self.target_sr = target_sr
        self.target_sr_se = target_sr_se
        self.audio_col = audio_col
        self.label_col = label_col
        self.resampler_cache = {}

        df_list = []
        for i in range(len(csv_path)):
            df_one = pd.read_csv(csv_path[i], usecols=[self.audio_col, self.label_col])
            df_list.append(df_one)

        self.df = pd.concat(df_list)

        self.audio_root_dir = audio_root_dir
        
        logging.info(f"From {csv_path} load {self.__len__()} samples. Audio root dir: {audio_root_dir}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item_row = self.df.iloc[idx]
        
        relative_path = item_row[self.audio_col]
        
        audio_path = os.path.join(self.audio_root_dir, relative_path)
        # print(audio_path)
        waveform_orig, original_sr = torchaudio.load(audio_path)

        # 重采样
        if original_sr != self.target_sr:
            resampler_name = str(original_sr)+str(self.target_sr)
            if resampler_name not in self.resampler_cache:
                self.resampler_cache[resampler_name] = torchaudio.transforms.Resample(original_sr, self.target_sr)
            waveform = self.resampler_cache[resampler_name](waveform_orig)
        else:
            waveform = waveform_orig

        # 转换为单声道
        if waveform.ndim > 1:
            waveform = torch.mean(waveform, dim=0)
        
        # 语义重采样
        if original_sr != self.target_sr_se:
            resampler_name = str(original_sr)+str(self.target_sr_se)
            if resampler_name not in self.resampler_cache:
                self.resampler_cache[resampler_name] = torchaudio.transforms.Resample(original_sr, self.target_sr_se)
            waveform_se = self.resampler_cache[resampler_name](waveform_orig)
        else:
            waveform_se = waveform_orig

        # 转换为单声道
        if waveform_se.ndim > 1:
            waveform_se = torch.mean(waveform_se, dim=0)

        wav, sr = librosa.load(audio_path, sr=16000)
        lps = torch.from_numpy(np.expand_dims(np.abs(librosa.stft(wav, n_fft = 512, hop_length=256,win_length=512)).T, axis=0))
        lps = lps.permute(1, 0, 2).contiguous()
        wav = torch.from_numpy(wav)

        # 获取标签
        labels = {}
        if self.label_col in item_row and pd.notna(item_row[self.label_col]):
            # 假设CSV中的标签列对应质量
            labels['quality'] = torch.tensor(item_row[self.label_col], dtype=torch.float32)
        
        return {"waveform": waveform, "waveform_se": waveform_se, "labels": labels, "sampling_rate": self.target_sr, "lps": lps, "wav": wav}

class MyCollator:
    def __init__(self, downsampling_factor, downsampling_factor_se):
        self.downsampling_factor = downsampling_factor
        self.downsampling_factor_se = downsampling_factor_se

    def __call__(self, batch):
        waveforms = [item['waveform'] for item in batch]
        waveforms_se = [item['waveform_se'] for item in batch]
        labels = [item['labels'] for item in batch]
        # ssl_input = [item['ssl_input'] for item in batch]
        wav = [item['wav'] for item in batch]
        lps = [item['lps'] for item in batch]
        
        waveform_lengths = [w.shape[-1] for w in waveforms]
        waveform_lengths_se = [w.shape[-1] for w in waveforms_se]
        token_lengths = [np.ceil(length / self.downsampling_factor) for length in waveform_lengths]
        token_lengths_se = [np.floor(length / self.downsampling_factor_se) for length in waveform_lengths_se]

        padded_waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True, padding_value=0.0)
        padded_waveforms_se = torch.nn.utils.rnn.pad_sequence(waveforms_se, batch_first=True, padding_value=0.0)
        padded_wav = torch.nn.utils.rnn.pad_sequence(wav, batch_first=True, padding_value=0.0)
        padded_lps = torch.nn.utils.rnn.pad_sequence(lps, batch_first=True, padding_value=0.0)
        padded_lps = padded_lps.permute(0, 2, 1, 3)
        
        collated_labels = {}
        if labels and labels[0]: 
            for key in labels[0].keys():
                if all(key in l for l in labels):
                    collated_labels[key] = torch.stack([l[key] for l in labels if key in l])

        return {"waveforms": padded_waveforms, "waveforms_se": padded_waveforms_se, "labels": collated_labels, 
                "wav": padded_wav, "lps": padded_lps,
                "token_lengths": torch.tensor(token_lengths, dtype=torch.long),
                "token_lengths_se": torch.tensor(token_lengths_se, dtype=torch.long)
                }

def collate_fn(batch):
    waveforms = [item['waveform'] for item in batch]
    labels = [item['labels'] for item in batch]
    whispers = [item['whisper'] for item in batch]

    padded_waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    padded_whispers = torch.nn.utils.rnn.pad_sequence(whispers, batch_first=True, padding_value=0.0)
    
    collated_labels = {}
    if labels and labels[0]:
        for key in labels[0].keys():
            if all(key in l for l in labels):
                collated_labels[key] = torch.stack([l[key] for l in labels if key in l])

    return {"waveforms": padded_waveforms, "labels": collated_labels, "whispers": padded_whispers}

def get_dataloader(dataset_path_or_name, split, target_sr, target_sr_se, batch_size, shuffle=True, audio_root_dir=None, collate_fn=collate_fn):
    
    dataset = None
    
    if dataset_path_or_name[0].endswith('.csv') or dataset_path_or_name.endswith('.csv'):
        if not audio_root_dir:
            raise ValueError("--audio_root_dir is required")
        if type(dataset_path_or_name) is not list:
            dataset_path_or_name = [dataset_path_or_name]
        logging.info(f"Loading data: {dataset_path_or_name}")
        dataset = CsvAudioDataset(
            csv_path=dataset_path_or_name, 
            audio_root_dir=audio_root_dir, 
            target_sr=target_sr,
            target_sr_se=target_sr_se,
            audio_col='filepath_deg',
            label_col='mos'
        )
    else:
        raise ValueError(f"Unsupported dataset name or invalid CSV path: {dataset_path_or_name}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=32)
    
    logging.info(f"DataLoader successfully created with a batch size of {batch_size}")
    return dataloader