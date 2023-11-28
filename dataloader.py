import numpy as np
import torch
import random
import torch.utils.data
import glob
import torchaudio
import torch.nn.functional as F

def vec_resampling(vec_src, dst_len):
    vec_src = F.interpolate(vec_src.unsqueeze(0), dst_len, mode='linear')
    return vec_src.squeeze()

class AudioDataLoader(torch.utils.data.Dataset):
    def __init__(self, path, hparams=None):
        self.hparams = hparams
        self.audiopaths = glob.glob(path+f'\\*\\*.wav', recursive=True)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.use_mel_spec_posterior = getattr(
            hparams, "use_mel_posterior_encoder", False
        )
        if self.use_mel_spec_posterior:
            self.n_mel_channels = getattr(hparams, "n_mel_channels", 80)
        self.add_blank = hparams.add_blank
        self.min_audio_len = getattr(hparams, "min_audio_len", 8192)
        self.sids = {}

        count = 0
        for a in self.audiopaths:
            tmp = a.split("\\")
            if self.sids.get(tmp[2]) is None:
                self.sids[tmp[2]] = count
                count += 1

        random.seed(7650)
        random.shuffle(self.audiopaths)

    def get_data(self, audiopath):
        tmp = audiopath.split("\\")
        sid = torch.LongTensor([self.sids[tmp[2]]])
        vec = torch.FloatTensor(np.load(audiopath.replace(".wav", ".vec.npy")))
        spec = torch.FloatTensor(np.load(audiopath.replace(".wav", ".spec.npy")))
        cqt = torch.FloatTensor(np.load(audiopath.replace(".wav", ".cqt.npy")))
        vec = vec_resampling(vec, spec.shape[1])
        cqt = vec_resampling(cqt, spec.shape[1])
        wav, sr = torchaudio.load(audiopath)
        if len(wav.shape) == 2 and wav.shape[1] >= 2:
            wav = torch.mean(wav, dim=0)
        assert sr == self.hparams.sampling_rate, "sampling rate error"
        wav_norm = torch.FloatTensor(wav)
        wav_norm = wav_norm.unsqueeze(0)

        return vec, spec, cqt, wav_norm, sid

    def __getitem__(self, index):
        return self.get_data(self.audiopaths[index])

    def __len__(self):
        return len(self.audiopaths)

class DataCollate:

    def __init__(self):
        pass

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size(1) for x in batch]),
            dim=0, descending=True)

        max_vec_len = max([x[0].size(1) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_cqt_len = max([x[2].size(1) for x in batch])
        max_wav_len = max([x[3].size(1) for x in batch])

        vec_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        cqt_lengths = torch.LongTensor(len(batch))
        lengths = vec_lengths
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        vec_padded = torch.FloatTensor(len(batch), batch[0][0].size(0), max_vec_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        cqt_padded = torch.FloatTensor(len(batch), batch[0][2].size(0), max_cqt_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)

        vec_padded.zero_()
        spec_padded.zero_()
        cqt_padded.zero_()
        wav_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            vec = row[0]
            vec_padded[i, :, : vec.size(1)] = vec
            vec_lengths[i] = vec.size(1)

            spec = row[1]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            cqt = row[2]
            cqt_padded[i, :, : cqt.size(1)] = cqt
            cqt_lengths[i] = cqt.size(1)

            wav = row[3]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[4]

        return (
            vec_padded,
            spec_padded,
            cqt_padded,
            lengths,
            wav_padded,
            wav_lengths,
            sid,
        )
