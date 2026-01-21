import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import torchaudio
import logging
import pandas as pd
import os
from transformers import AutoFeatureExtractor

# Tokenizer的降采样倍数
# DOWNSAMPLING_FACTOR = 320

# 数据集在 Hugging Face Hub 上的占位符名称
DATASET_MAPPING = {
    "nisqa": "/data/home/wangchaoyang/database/NISQA_Corpus/NISQA_TRAIN_SIM/NISQA_TRAIN_SIM_file.csv",
    "bvcc": "fx-wood/voice-mos-challenge-2022-bvcc-track1",
    "somos": "speech-somos/somos",
    "tmhint_qi": "dhimasryan/tmhint_qi_voicemos2023"
}

# 标签列的映射
LABEL_MAPPING = {
    "bvcc": {"quality": "mos", "intelligibility": None},
    "somos": {"quality": "mos", "intelligibility": None},
    "tmhint_qi": {"quality": "quality", "intelligibility": "intelligibility"}
}

class CsvAudioDataset(Dataset):
    """一个用于从CSV文件加载音频和标签的通用数据集类。"""
    def __init__(self, csv_path, audio_root_dir, target_sr, target_sr_se, audio_col='filepath_deg', label_col='mos'):
        """
        Args:
            csv_path (str): CSV文件的路径。
            audio_root_dir (str): 音频文件所在的根目录。
            target_sr (int): 目标采样率。
            audio_col (str): CSV中包含音频相对路径的列名。
            label_col (str): CSV中包含标签的列名。
        """
        self.target_sr = target_sr
        self.target_sr_se = target_sr_se
        self.audio_col = audio_col
        self.label_col = label_col
        self.resampler_cache = {}
        # self.ssl_input_sr = ssl_input_sr

        # assert len(csv_path) == len(whisper_root_dir)

        df_list = []
        for i in range(len(csv_path)):
            df_one = pd.read_csv(csv_path[i], usecols=[self.audio_col, self.label_col])
            df_list.append(df_one)

        self.df = pd.concat(df_list)

        self.audio_root_dir = audio_root_dir
        # self.whisper_list = open(whisper_txt, 'r').read().splitlines()
        
        
        logging.info(f"从 {csv_path} 加载了 {self.__len__()} 个样本。音频根目录: {audio_root_dir}")

        # self.ssl_featex = AutoFeatureExtractor.from_pretrained('/data/home/wangchaoyang/.cache/huggingface/hub/models--facebook--wav2vec2-large-960h-lv60/snapshots/8e7d14742e8f98c6bbb24e5231406af321a8f9ce')

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
    
    def get_npy_path_from_audio(self, audio_path, npy_directory):
        """
        根据音频文件的路径，获取在另一个目录中具有相同主干名称的.npy文件的完整路径。

        Args:
            audio_path (str): 原始音频文件的完整路径。
            npy_directory (str): 存放.npy文件的目录路径。

        Returns:
            str: 对应的.npy文件的完整路径。
        """
        # 1. 从音频路径中提取完整的文件名 (例如: "c00001_3_640_2_7_001-ch6-speaker_seg49.wav")
        audio_filename_with_ext = os.path.basename(audio_path)
        
        # 2. 将完整文件名分割为主干名称和扩展名
        #    (例如: "c00001_3_640_2_7_001-ch6-speaker_seg49" 和 ".wav")
        audio_filename_base, _ = os.path.splitext(audio_filename_with_ext)
        
        # 3. 为主干名称添加新的.npy扩展名
        npy_filename = audio_filename_base + ".npy"
        
        # 4. 将.npy文件名与目标目录路径拼接起来，形成最终的完整路径
        full_npy_path = os.path.join(npy_directory, npy_filename)
        
        return full_npy_path

class MyCollator:
    # 1. 在初始化方法中接收 DOWNSAMPLING_FACTOR
    def __init__(self, downsampling_factor, downsampling_factor_se):
        self.downsampling_factor = downsampling_factor
        self.downsampling_factor_se = downsampling_factor_se
    # 2. 将原来的 collate_fn 逻辑放入 __call__ 方法中
    #    这样类的实例就可以像函数一样被调用
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
        # padded_ssl_input = torch.nn.utils.rnn.pad_sequence(ssl_input, batch_first=True, padding_value=0.0)
        padded_wav = torch.nn.utils.rnn.pad_sequence(wav, batch_first=True, padding_value=0.0)
        padded_lps = torch.nn.utils.rnn.pad_sequence(lps, batch_first=True, padding_value=0.0)
        padded_lps = padded_lps.permute(0, 2, 1, 3)
        # padded_lps = 0
        
        # 将标签字典列表转换为字典，其中每个键对应一个批次的张量
        collated_labels = {}
        if labels and labels[0]: # 确保列表不为空且第一个元素不为空
            for key in labels[0].keys():
                # 检查批次中的所有字典是否都包含此键
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
    # ssl_input = [item['ssl_input'] for item in batch]

    padded_waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    padded_whispers = torch.nn.utils.rnn.pad_sequence(whispers, batch_first=True, padding_value=0.0)
    # padded_ssl_input = torch.nn.utils.rnn.pad_sequence(ssl_input, batch_first=True, padding_value=0.0)
    
    # 将标签字典列表转换为字典，其中每个键对应一个批次的张量
    collated_labels = {}
    if labels and labels[0]: # 确保列表不为空且第一个元素不为空
        for key in labels[0].keys():
            # 检查批次中的所有字典是否都包含此键
            if all(key in l for l in labels):
                collated_labels[key] = torch.stack([l[key] for l in labels if key in l])

    return {"waveforms": padded_waveforms, "labels": collated_labels, "whispers": padded_whispers}

def get_dataloader(dataset_path_or_name, split, target_sr, target_sr_se, batch_size, shuffle=True, audio_root_dir=None, collate_fn=collate_fn):
    """加载并返回指定数据集的 DataLoader"""
    
    dataset = None
    
    # 新增逻辑：如果 dataset_path_or_name 是一个 .csv 文件路径，则使用 CsvAudioDataset
    if dataset_path_or_name[0].endswith('.csv') or dataset_path_or_name.endswith('.csv'):
        if not audio_root_dir:
            raise ValueError("使用CSV数据集时，必须提供 --audio_root_dir 参数。")
        if type(dataset_path_or_name) is not list:
            dataset_path_or_name = [dataset_path_or_name]
        logging.info(f"正在加载本地CSV数据集: {dataset_path_or_name}")
        # 假设CSV用于质量评估，标签列为 'mos'，音频路径列为 'filepath_deg'
        dataset = CsvAudioDataset(
            csv_path=dataset_path_or_name, 
            audio_root_dir=audio_root_dir, 
            target_sr=target_sr,
            target_sr_se=target_sr_se,
            audio_col='filepath_deg',
            label_col='mos'
        )
    else:
        raise ValueError(f"不支持的数据集名称或无效的CSV路径: {dataset_path_or_name}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=32)
    
    logging.info(f"成功创建 DataLoader，批大小为 {batch_size}")
    return dataloader