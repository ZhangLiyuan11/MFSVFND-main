import os
import pandas as pd

from torch.utils.data import Dataset
import torch
import numpy as np

class MFSVFNDDataset(Dataset):
    def __init__(self, path_vid, token, dataset, datamode='title+ocr'):
        self.dataset = dataset
        self.vid = []
        self.tokenizer = token
        self.datamode = datamode

        if self.dataset == 'fakesv':
            self.data_complete = pd.read_json('data/FakeSV/data_complete.json', orient='records', dtype=False,
                                              lines=True)
            self.data_complete = self.data_complete[self.data_complete['annotation'] != '辟谣']
            self.maefeapath = 'data/fea/fakesv/mae_fea'
            self.hubert_path = 'data/fea/fakesv/hubert_fea/'
            with open('data/FakeSV/data-split/' + path_vid, "r") as fr:
                for line in fr.readlines():
                    self.vid.append(line.strip())
        else:
            self.data_complete = pd.read_json('data/FakeTT/data.json', orient='records', dtype=False, lines=True)
            self.maefeapath = 'data/fea/fakett/mae_fea'
            self.hubert_path = 'data/fea/fakett/hubert_fea/'
            with open('data/FakeTT/data-split/' + path_vid, "r") as fr:
                for line in fr.readlines():
                    self.vid.append(line.strip())

        self.data = self.data_complete[self.data_complete.video_id.isin(self.vid)]
        self.data['video_id'] = self.data['video_id'].astype('category')
        self.data['video_id'].cat.set_categories(self.vid)
        self.data.sort_values('video_id', ascending=True, inplace=True)
        self.data.reset_index(inplace=True)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']

        # label
        if self.dataset == 'fakesv':
            # label
            label = 0 if item['annotation'] == '真' else 1
            # text
            if self.datamode == 'title+ocr':
                title_tokens = self.tokenizer(item['title'] + ' ' + item['ocr'] + '' + item['keywords'], max_length=512,
                                              padding='max_length', truncation=True)
            else:
                title_tokens = self.tokenizer(item['title'] + '' + item['keywords'], max_length=512,
                                              padding='max_length', truncation=True)

        else:
            label = 0 if item['annotation'] == 'real' else 1
            # text
            if self.datamode == 'title+ocr':
                title_tokens = self.tokenizer(item['description'] + ' ' + item['recognize_ocr'] + '' + item['event'],
                                              max_length=512, padding='max_length', truncation=True)
            else:
                title_tokens = self.tokenizer(item['recognize_ocr'] + '' + item['event'], max_length=512,
                                              padding='max_length', truncation=True)

        # label
        label = torch.tensor(label)

        # text
        title_inputid = torch.LongTensor(title_tokens['input_ids'])
        title_mask = torch.LongTensor(title_tokens['attention_mask'])

        # audio
        audio_item_path = self.hubert_path + vid + '.pkl'
        audio_fea = torch.load(audio_item_path)

        # frames
        file_path = os.path.join(self.maefeapath, vid + '.pkl')
        f = open(file_path, 'rb')
        frames = torch.load(f, map_location='cpu')
        frames = torch.FloatTensor(frames)

        return {
            'label': label,
            'title_inputid': title_inputid,
            'title_mask': title_mask,
            'audio_fea': audio_fea,
            'frames': frames,
        }


def pad_frame_sequence_fakesv(seq_len, lst, modality):
    attention_masks = []
    result = []
    for item in lst:
        if modality == 'video':
            if len(item.shape) == 1:
                item = item.unsqueeze(0)
        else:
            item = torch.squeeze(item)
        item = torch.FloatTensor(item)
        ori_len = item.shape[0]
        if ori_len >= seq_len:
            gap = ori_len // seq_len
            item = item[::gap][:seq_len]
            mask = np.ones((seq_len))
        else:
            item = torch.cat((item, torch.zeros([seq_len - ori_len, item.shape[1]], dtype=torch.float)), dim=0)
            mask = np.append(np.ones(ori_len), np.zeros(seq_len - ori_len))
        result.append(item)
        mask = torch.IntTensor(mask)
        attention_masks.append(mask)
    return torch.stack(result), torch.stack(attention_masks)


def pad_frame_sequence_fakett(seq_len, lst):
    attention_masks = []
    result = []
    for video in lst:
        video = torch.squeeze(video)
        video = torch.FloatTensor(video)
        ori_len = video.shape[0]
        if ori_len >= seq_len:
            gap = ori_len // seq_len
            video = video[::gap][:seq_len]
            mask = np.ones((seq_len))
        else:
            video = torch.cat((video, torch.zeros([seq_len - ori_len, video.shape[1]], dtype=torch.float)), dim=0)
            mask = np.append(np.ones(ori_len), np.zeros(seq_len - ori_len))
        result.append(video)
        mask = torch.IntTensor(mask)
        attention_masks.append(mask)
    return torch.stack(result), torch.stack(attention_masks)


def fakesv_collate_fn(batch):
    num_frames = 86
    num_audioframes = 80

    title_inputid = [item['title_inputid'] for item in batch]
    title_mask = [item['title_mask'] for item in batch]

    # pad video frames
    frames = [item['frames'] for item in batch]
    frames, frames_masks = pad_frame_sequence_fakesv(num_frames, frames, modality='video')

    # pad audio frames
    audio_feas = [item['audio_fea'] for item in batch]
    audio_feas, audiofeas_masks = pad_frame_sequence_fakesv(num_audioframes, audio_feas, modality='audio')

    label = [item['label'] for item in batch]

    return {
        'label': torch.stack(label),
        'title_inputid': torch.stack(title_inputid),
        'title_mask': torch.stack(title_mask),
        'audio_feas': audio_feas,
        'audiofeas_masks': audiofeas_masks,
        'frames': frames,
        'frames_masks': frames_masks,
    }


def fakett_collate_fn(batch):
    num_frames = 111
    num_audioframes = 103

    title_inputid = [item['title_inputid'] for item in batch]
    title_mask = [item['title_mask'] for item in batch]

    frames = [item['frames'] for item in batch]
    frames, frames_masks = pad_frame_sequence_fakett(num_frames, frames)

    audio_feas = [item['audio_fea'] for item in batch]
    audio_feas, audiofeas_masks = pad_frame_sequence_fakett(num_audioframes, audio_feas)

    label = [item['label'] for item in batch]

    return {
        'label': torch.stack(label),
        'title_inputid': torch.stack(title_inputid),
        'title_mask': torch.stack(title_mask),
        'audio_feas': audio_feas,
        'audiofeas_masks': audiofeas_masks,
        'frames': frames,
        'frames_masks': frames_masks,
    }
