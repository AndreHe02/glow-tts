import matplotlib.pyplot as plt
import IPython.display as ipd

import sys
sys.path.append('./waveglow/')

import librosa
import numpy as np
import os
import glob
import json

import torch
from text import text_to_sequence, cmudict
from text.symbols import symbols
import commons
import attentions
import modules
import models
import utils

from lm_scorer.models.auto import AutoLMScorer as LMScorer

# load WaveGlow
waveglow_path = './waveglow/waveglow_256channels_universal_v5.pt' # or change to the latest version of the pretrained WaveGlow.
waveglow = torch.load(waveglow_path)['model']
waveglow = waveglow.remove_weightnorm(waveglow)
_ = waveglow.cuda().eval()


hps = utils.get_hparams_from_file("./configs/base.json")
checkpoint_path = "pretrained/pretrained.pth"

model = models.FlowGenerator(
    len(symbols) + getattr(hps.data, "add_blank", False),
    out_channels=hps.data.n_mel_channels,
    **hps.model).to("cuda")

utils.load_checkpoint(checkpoint_path, model)
model.decoder.store_inverse() # do not calcuate jacobians for fast decoding
_ = model.eval()

cmu_dict = cmudict.CMUDict(hps.data.cmudict_path)

print(cmu_dict)

# normalizing & type casting
def normalize_audio(x, max_wav_value=hps.data.max_wav_value):
    return np.clip((x / np.abs(x).max()) * max_wav_value, -32768, 32767).astype("int16")

sents = []
phone_durs = {}
with open("timit.txt", "r") as infile:
    for line in infile.readlines():
        text = " ".join(line.split(",")[1:])
        sents.append(text)

sent_durs = []
for sent_idx, sent in enumerate(sents):
    tst_stn = sent
    if getattr(hps.data, "add_blank", False):
        text_norm = text_to_sequence(tst_stn.strip(), ['english_cleaners'], cmu_dict)
        text_norm = commons.intersperse(text_norm, len(symbols))
    else: # If not using "add_blank" option during training, adding spaces at the beginning and the end of utterance improves quality
        tst_stn = " " + tst_stn.strip() + " "
        text_norm = text_to_sequence(tst_stn.strip(), ['english_cleaners'], cmu_dict)

    for word in tst_stn.split():
        seq = text_to_sequence(word, ['english_cleaners'], cmu_dict)
        # print(word, seq)

    sequence = np.array(text_norm)[None, :]
    # print("".join([symbols[c] if c < len(symbols) else "<BNK>" for c in sequence[0]]))
    x_tst = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    x_tst_lengths = torch.tensor([x_tst.shape[1]]).cuda()
    
    # print(x_tst.detach().numpy()[0][0])

    with torch.no_grad():
      noise_scale = .667
      length_scale = 1.0
      (y_gen_tst, *_), *_, (attn_gen, durs, _) = model(x_tst, x_tst_lengths, gen=True, noise_scale=noise_scale, length_scale=length_scale)
      durs = durs.detach().cpu().numpy()[0][0]
      sent_durs.append(durs)
      for phone, dur in zip(list(sequence[0]), list(durs)):
          if phone in phone_durs:
              phone_durs[phone].append(dur)
          else:
              phone_durs[phone] = [dur]
      

phone_vars = {}
for phone in phone_durs:
    phone_vars[phone] = np.var(np.asarray(phone_durs[phone]))

device = "cuda:0" if torch.cuda.is_available() else "cpu"
batch_size = 1
scorer = LMScorer.from_pretrained("gpt2", device=device, batch_size=batch_size)

for sent_idx, sent in enumerate(sents):
    tst_stn = sent
    if getattr(hps.data, "add_blank", False):
        text_norm = text_to_sequence(tst_stn.strip(), ['english_cleaners'], cmu_dict)
        text_norm = commons.intersperse(text_norm, len(symbols))
    else: # If not using "add_blank" option during training, adding spaces at the beginning and the end of utterance improves quality
        tst_stn = " " + tst_stn.strip() + " "
        text_norm = text_to_sequence(tst_stn.strip(), ['english_cleaners'], cmu_dict)
    print(sent)
    words = []
    word_lengths = []
    for word in tst_stn.split():
        seq = text_to_sequence(word, ['english_cleaners'], cmu_dict)
        words.append(word)
        word_lengths.append(len(seq))
        # print(word, seq)
    word_map = []
    phone_counter = 0
    for word_idx, length in enumerate(word_lengths):
        for phone_idx in range(length):
            word_map.append(word_idx)
        word_map.append(-1)
    word_map = word_map[:-1]
    # print(word_map)
    # print(len(word_map), sum(word_lengths))


    sequence = np.array(text_norm)[None, :]
    # print("".join([symbols[c] if c < len(symbols) else "<BNK>" for c in sequence[0]]))
    x_tst = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    x_tst_lengths = torch.tensor([x_tst.shape[1]]).cuda()
    # print(x_tst)
    # print(x_tst.shape)

    score = scorer.tokens_score(sent, log=True)
    # print(score)
    log_probs = score[0]
    subwords = score[2]
    num_words = len(tst_stn.split())
    print(tst_stn.split())
    word_score = np.zeros(num_words)
    x = sum([word[0] == "Ġ" for word in subwords])+1
    word_idx = 0
    subword_idx = 0
    while (subword_idx < len(subwords)-1):
        if subwords[subword_idx][0] == "Ġ" and len(subwords[subword_idx]) > 1:
            word_idx += 1
        word_score[word_idx] += log_probs[subword_idx]
        subword_idx += 1
    print(word_score)
    dur = sent_durs[sent_idx]
    for word_idx in range(len(list(word_score))):
        val = np.abs(word_score[word_idx])
        if val > 10.0:
            for phone_idx, value in enumerate(word_map):
                if value == word_idx:
                    dur[phone_idx] += 2.0 * phone_vars[sequence[0][phone_idx]]

    #dur = sent_durs[sent_idx]
    #for idx in range(len(list(dur))):
    #    dur[idx] += 2.0 * phone_vars[sequence[0][idx]]
    hardcoded_dur = torch.tensor([[dur]]).cuda()
    # print(x_tst.detach().numpy()[0][0])

    with torch.no_grad():
      noise_scale = .667
      length_scale = 1.0
      (y_gen_tst, *_), *_, (attn_gen, durs, _) = model(x_tst, x_tst_lengths, gen=True, noise_scale=noise_scale, length_scale=length_scale, hardcoded_durs=hardcoded_dur)
      durs = durs.detach().cpu().numpy()[0][0]      
      try:
        audio = waveglow.infer(y_gen_tst.half(), sigma=.666)
      except:
        audio = waveglow.infer(y_gen_tst, sigma=.666)
    audio = ipd.Audio(normalize_audio(audio[0].clamp(-1,1).data.cpu().float().numpy()), rate=hps.data.sampling_rate)
    with open("/home/nickatomlin/andrehe/tmp/timit-{}.wav".format(sent_idx), "wb") as infile:
        infile.write(audio.data)




