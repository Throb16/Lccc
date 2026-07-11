import json
import os
from random import randint

import numpy as np
from imageio import imread
from torch.utils.data import Dataset

from preprocess_data import encode


class WHUCDCDataset(Dataset):
    """
    WHU-CDC dataloader with the same return format as LEVIRCCDataset:
    imgA, imgB, seg_label, token_all, token_all_len, token, token_len, name.
    """

    def __init__(self, data_folder, list_path, split, token_folder=None,
                 vocab_file=None, max_length=41, allow_unk=0, max_iters=None):
        self.mean = [0.39073 * 255, 0.38623 * 255, 0.32989 * 255]
        self.std = [0.15329 * 255, 0.14628 * 255, 0.13648 * 255]
        self.list_path = list_path
        self.split = split
        self.max_length = max_length
        self.allow_unk = allow_unk

        assert self.split in {'train', 'val', 'test'}

        list_file = os.path.join(list_path, split + '.txt')
        self.img_ids = [i.strip() for i in open(list_file, encoding='utf-8') if i.strip()]

        self.word_vocab = None
        if vocab_file is not None:
            with open(os.path.join(list_path, vocab_file + '.json'), 'r', encoding='utf-8') as f:
                self.word_vocab = json.load(f)

        if max_iters is not None:
            n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
            self.img_ids = (self.img_ids * n_repeat)[:max_iters]

        self.files = []
        for name in self.img_ids:
            image_name = name.split('-')[0]
            token_id = name.split('-')[-1] if '-' in name else None
            stem = os.path.splitext(name)[0]
            token_file = os.path.join(token_folder, stem + '.txt') if token_folder is not None else None

            self.files.append({
                "imgA": os.path.join(data_folder, split, 'A', image_name),
                "imgB": os.path.join(data_folder, split, 'B', image_name),
                "seg_label": os.path.join(data_folder, split, 'label', image_name),
                "token": token_file,
                "token_id": token_id,
                "name": image_name,
            })

    def __len__(self):
        return len(self.files)

    def _read_rgb(self, path):
        img = imread(path)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.shape[2] == 4:
            img = img[..., :3]
        return img

    def _read_label(self, path):
        label = imread(path)
        if label.ndim == 3:
            label = label[..., 0]
        label = np.asarray(label)
        return (label > 0).astype(np.int64)

    def __getitem__(self, index):
        datafiles = self.files[index]
        name = datafiles["name"]

        imgA = np.asarray(self._read_rgb(datafiles["imgA"]), np.float32).transpose(2, 0, 1)
        imgB = np.asarray(self._read_rgb(datafiles["imgB"]), np.float32).transpose(2, 0, 1)
        seg_label = self._read_label(datafiles["seg_label"])

        for i in range(len(self.mean)):
            imgA[i, :, :] -= self.mean[i]
            imgA[i, :, :] /= self.std[i]
            imgB[i, :, :] -= self.mean[i]
            imgB[i, :, :] /= self.std[i]

        if datafiles["token"] is not None:
            with open(datafiles["token"], encoding='utf-8') as caption:
                caption_list = json.loads(caption.read())

            token_all = np.zeros((len(caption_list), self.max_length), dtype=int)
            token_all_len = np.zeros((len(caption_list), 1), dtype=int)
            for j, tokens in enumerate(caption_list):
                nochange_cap = ['<START>', 'the', 'scene', 'is', 'the', 'same', 'as', 'before', '<END>']
                if self.split == 'train' and nochange_cap in caption_list:
                    tokens = nochange_cap
                tokens_encode = encode(tokens, self.word_vocab, allow_unk=self.allow_unk == 1)
                length = min(len(tokens_encode), self.max_length)
                token_all[j, :length] = tokens_encode[:length]
                token_all_len[j] = length

            if datafiles["token_id"] is not None:
                token_index = int(datafiles["token_id"])
                token = token_all[token_index]
                token_len = token_all_len[token_index].item()
            else:
                token_index = randint(0, len(caption_list) - 1)
                token = token_all[token_index]
                token_len = token_all_len[token_index].item()
        else:
            token_all = np.zeros((1, self.max_length), dtype=int)
            token = np.zeros((self.max_length,), dtype=int)
            token_len = 0
            token_all_len = np.zeros((1, 1), dtype=int)

        return (
            imgA.copy(), imgB.copy(), seg_label.copy(),
            token_all.copy(), token_all_len.copy(), token.copy(),
            np.array(token_len), name
        )
