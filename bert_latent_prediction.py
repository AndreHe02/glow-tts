import utils
import models
import re
import os
import torch
import torch.nn as nn
import torch.optim as optim
import commons
import random
import numpy as np
import transformers

from transformers import DistilBertModel, DistilBertTokenizer, DistilBertConfig
from torch.utils.data import DataLoader
from text.symbols import symbols
from data_utils import TextMelLoader, TextMelCollate
from text import _clean_text, text_to_sequence, cmudict
from tqdm import tqdm


class WordPhoneMelLoader(TextMelLoader):

    def __getitem__(self, index):
        audiopath, sent = self.audiopaths_and_text[index]
        phones, mel = self.get_mel_text_pair((audiopath, sent))

        clean_sent = _clean_text(sent, ['english_cleaners'])
        wordpieces = tokenizer.encode(clean_sent, add_special_tokens=True)
        wordpieces = torch.IntTensor(wordpieces)

        words = clean_sent.split(" ")
        wordpiece_attn = torch.zeros((len(wordpieces), len(words)))
        phone_attn = torch.zeros((len(phones), len(words)))

        wp_idx = 0
        ph_idx = 0
        wordpieces_ = wordpieces.numpy()
        phones_ = phones.numpy()

        for i, word in enumerate(words):
            phs = text_to_sequence(word, ['english_cleaners'], cmu_dict)
            wps = tokenizer.encode(word, add_special_tokens=False)

            while np.any(wordpieces_[wp_idx:wp_idx + len(wps)] - wps):
                if wp_idx + len(wps) >= len(wordpieces_):
                    break
                wp_idx += 1
            if not np.any(wordpieces_[wp_idx:wp_idx + len(wps)] - wps):
                wordpiece_attn[wp_idx:wp_idx + len(wps), i] = 1

            while np.any(phones_[ph_idx:ph_idx + len(phs)] - phs):
                if ph_idx + len(phs) >= len(phones_):
                    break
                ph_idx += 1
            if not np.any(phones_[ph_idx:ph_idx + len(phs)] - phs):
                phone_attn[ph_idx:ph_idx + len(phs), i] = 1
                if ph_idx + len(phs) < len(phones_) and phones_[ph_idx + len(phs)] == 11:
                    phone_attn[ph_idx + len(phs), i] = 1
                    ph_idx += 1

        assert torch.all(wordpiece_attn.sum(dim=0))
        assert torch.all(phone_attn.sum(dim=0))

        return wordpieces, phones, mel, wordpiece_attn, phone_attn


class WordPhoneMelCollate(TextMelCollate):

    def __call__(self, batch):
        text_lengths = torch.LongTensor([len(x[0]) for x in batch])
        max_text_len = max(text_lengths)
        text_padded = torch.LongTensor(len(batch), max_text_len)
        text_padded.zero_()
        for i in range(len(batch)):
            text = batch[i][0]
            text_padded[i, :text.size(0)] = text

        phone_lengths = torch.LongTensor([len(x[1]) for x in batch])
        max_phone_len = max(phone_lengths)
        phones_padded = torch.LongTensor(len(batch), max_phone_len)
        phones_padded.zero_()
        for i in range(len(batch)):
            phones = batch[i][1]
            phones_padded[i, :phones.size(0)] = phones

        num_mels = batch[0][2].size(0)
        max_target_len = max([x[2].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        mel_lengths = torch.LongTensor(len(batch))
        for i in range(len(batch)):
            mel = batch[i][2]
            mel_padded[i, :, :mel.size(1)] = mel
            mel_lengths[i] = mel.size(1)

        max_word_count = max([x[3].size(1) for x in batch])
        wordpiece_attn_padded = torch.zeros((len(batch), max_text_len, max_word_count))
        phone_attn_padded = torch.zeros((len(batch), max_phone_len, max_word_count))
        for i in range(len(batch)):
            wordpiece_attn_padded[i, :batch[i][3].size(0), :batch[i][3].size(1)] = batch[i][3]
            phone_attn_padded[i, :batch[i][4].size(0), :batch[i][4].size(1)] = batch[i][4]

        return text_padded, text_lengths, phones_padded, phone_lengths, mel_padded, mel_lengths, wordpiece_attn_padded, phone_attn_padded


class MemLoader:
    ITEMS = ['wp', 'wp_len', 'ph', 'ph_len', 'mel_', 'mel_len', 'wp_attn', 'ph_attn']
    dummy = 'wp_attn'

    def __init__(self, path):
        self.path = path
        filenames = os.listdir(path)
        self.filenames = [f for f in filenames if self.dummy in f]
        self.data = []
        for f in self.filenames:
            items = [torch.from_numpy(np.load(os.path.join(self.path, f.replace(self.dummy, item_name)))) for item_name in self.ITEMS]
            self.data.append(items)

    def __len__(self):
        return len(self.filenames)

    def shuffle(self):
        random.shuffle(self.data)

    def loader(self):
        yield from self.data




class BertLinearLP(nn.Module):

    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.dim = self.bert.config.dim
        self.ph_projs = nn.Embedding(len(symbols) + 1, self.dim * 80)

    def forward(self, wp, ph, wp_attn, ph_attn, attn):
        wp_embed = self.bert(wp)[0]  # [b, #wp, dim=768]
        word_embed = torch.einsum('bpd, bpw -> bwd', wp_embed, wp_attn)
        wp_per_word = torch.sum(wp_attn, dim=1).unsqueeze(dim=-1)
        word_embed = word_embed / torch.maximum(wp_per_word, torch.ones_like(wp_per_word))  # [b, #w, dim]
        ph_embed = torch.einsum('bwd, bpw -> bpd', word_embed, ph_attn)  # [b, #ph, dim]

        ph_proj = self.ph_projs(ph)  # [b, #ph, dim*80]
        ph_proj_mats = torch.reshape(ph_proj, (ph_proj.shape[0], ph_proj.shape[1], self.dim, 80))
        x = torch.einsum('bpd, bpdl -> blp', ph_embed, ph_proj_mats)
        z = torch.matmul(attn.squeeze(1).transpose(1, 2), x.transpose(1, 2)).transpose(1, 2)
        return z



def train(num_epochs):
    model.train()
    log = {'train': [], 'test': []}
    for epoch in range(num_epochs):
        total_loss = 0
        mem_train_dataset.shuffle()
        for batch_idx, (wp, wp_len, ph, ph_len, mel, mel_len, wp_attn, ph_attn) in tqdm(enumerate(mem_train_dataset.loader())):
            with torch.no_grad():
                wp, wp_len = wp.cuda(), wp_len.cuda()
                ph, ph_len = ph.cuda(), ph_len.cuda()
                mel, mel_len = mel.cuda(), mel_len.cuda()
                wp_attn, ph_attn = wp_attn.float().cuda(), ph_attn.float().cuda()  # [b, #wp, #wrds], [b, #ph, #wrds]

                (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_) = \
                    generator(ph, ph_len, mel, mel_len, gen=False)

            z_bert = model(wp, ph, wp_attn, ph_attn, attn)
            # loss = commons.mle_loss(z, z_m+z_bert, z_logs, logdet, z_mask)
            loss = commons.mle_loss(z, z_bert, z_logs, logdet, z_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(mem_train_dataset)
        test_loss = evaluate()
        print('Train, Test:')
        print(train_loss, test_loss)
        if len(log['test']) == 0 or test_loss < min(log['test']):
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       os.path.join(log_dir, 'ckpt'))
        log['train'].append(train_loss)
        log['test'].append(test_loss)
    np.save('train_loss', np.array(log['train']))
    np.save('test_loss', np.array(log['test']))
    return log


def evaluate():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (wp, wp_len, ph, ph_len, mel, mel_len, wp_attn, ph_attn) in enumerate(mem_test_dataset.loader()):
            wp, wp_len = wp.cuda(), wp_len.cuda()
            ph, ph_len = ph.cuda(), ph_len.cuda()
            mel, mel_len = mel.cuda(), mel_len.cuda()
            wp_attn, ph_attn = wp_attn.float().cuda(), ph_attn.float().cuda()  # [b, #wp, #wrds], [b, #ph, #wrds]

            (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_) = \
                generator(ph, ph_len, mel, mel_len, gen=False)

            z_bert = model(wp, ph, wp_attn, ph_attn, attn)
            # loss = commons.mle_loss(z, z_m+z_bert, z_logs, logdet, z_mask)
            loss = commons.mle_loss(z, z_bert, z_logs, logdet, z_mask)

            total_loss += loss.item()

    return total_loss / len(mem_test_dataset)


import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

hps = utils.get_hparams_from_file("./configs/base.json")
cmu_dict = cmudict.CMUDict(hps.data.cmudict_path)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

tune_files = hps.data.training_files
eval_files = 'filelists/ljs_audio_text_test_filelist.txt'

#collate_fn = WordPhoneMelCollate(1)
#tune_dataset = WordPhoneMelLoader(tune_files, hps.data)
#tune_loader = DataLoader(tune_dataset, num_workers=32, shuffle=True,
#                          batch_size=hps.train.batch_size, pin_memory=True,
#                          drop_last=True, collate_fn=collate_fn)
#eval_dataset = WordPhoneMelLoader(eval_files, hps.data)
#eval_loader = DataLoader(eval_dataset, num_workers=32, shuffle=False,
#                         batch_size=hps.train.batch_size, pin_memory=True,
#                         drop_last=False, collate_fn=collate_fn)

mem_train_dataset = MemLoader('temp_data/train/')
mem_test_dataset = MemLoader('temp_data/test/')

checkpoint_path = "pretrained/pretrained.pth"
generator = models.FlowGenerator(
    len(symbols) + getattr(hps.data, "add_blank", False),
    out_channels=hps.data.n_mel_channels,
    **hps.model).to("cuda")

utils.load_checkpoint(checkpoint_path, generator)
generator.decoder.store_inverse()  # do not calcuate jacobians for fast decoding
_ = generator.eval()

log_dir = 'pretrained_latent_non_ensemble'
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)
bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
model = BertLinearLP(bert).cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-5)
train(1)

log_dir = 'untrained_latent_non_ensemble'
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)
untrained_bert = DistilBertModel(DistilBertConfig())
model = BertLinearLP(untrained_bert).cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

train(1)


