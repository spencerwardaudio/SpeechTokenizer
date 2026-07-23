from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import random
import torch
import numpy as np

def collate_fn(data):
    # return pad_sequence(data, batch_first=True)
    # return pad_sequence(*data)
    is_one_data = not isinstance(data[0], tuple)
    outputs = []
    if is_one_data:
        for datum in data:
            if isinstance(datum, torch.Tensor):
                output = datum.unsqueeze(0)
            else:
                output = torch.tensor([datum])
            outputs.append(output)
        return tuple(outputs)        
    for datum in zip(*data):
        if isinstance(datum[0], torch.Tensor):
            output = pad_sequence(datum, batch_first=True)
        else:
            output = torch.tensor(list(datum))
        outputs.append(output)

    return tuple(outputs)

def get_dataloader(ds, **kwargs):
    return DataLoader(ds, collate_fn=collate_fn, **kwargs)

class audioDataset(Dataset):
    
    def __init__(self,
                 file_list,
                 segment_size,
                 sample_rate,
                 downsample_rate = 320,
                 valid=False):
        super().__init__()
        self.file_list = file_list
        self.segment_size = segment_size
        self.sample_rate = sample_rate
        self.valid = valid
        self.downsample_rate = downsample_rate
        
    def __len__(self):
        # Limit to 1010 steps × 8 batch = 8,080 samples/epoch (fast training mode)
        return min(len(self.file_list), 8080)
    
    
    def __getitem__(self, index):
        file = self.file_list[index].strip()
        
        # Check if file has features (speech with TAB) or just audio (FSD50K)
        if '\t' in file:
            audio_file, feature_file = file.split('\t')
            feature = torch.from_numpy(np.load(feature_file))
        else:
            audio_file = file
            feature = None  # Will create dummy features when distill_loss_lambda=0
        
        audio, sr = torchaudio.load(audio_file)
        audio = audio.mean(axis=0)
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
        
        if audio.size(-1) > self.segment_size:
            if self.valid:
                audio_segment = audio[:self.segment_size]
                if feature is not None:
                    feature_segment = feature[:self.segment_size // self.downsample_rate]
                else:
                    feature_segment = None  # Skip semantic features when not needed (distill_loss_lambda=0)
            else:
                max_audio_start = audio.size(-1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                audio_segment = audio[audio_start:audio_start+self.segment_size]
                
                if feature is not None:
                    feature_start = min(int(audio_start / self.downsample_rate), 
                                      feature.size(0) - self.segment_size // self.downsample_rate)
                    feature_segment = feature[feature_start:feature_start + self.segment_size // self.downsample_rate, :]
                else:
                    feature_segment = None  # Skip semantic features when not needed (distill_loss_lambda=0)
        else:
            if not self.valid:
                audio_segment = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(-1)), 'constant')
            else:
                audio_segment = audio
            
            if feature is None:
                feature_segment = None  # Skip semantic features when not needed (distill_loss_lambda=0)
            else:
                feature_segment = feature
        
        return audio_segment, feature_segment