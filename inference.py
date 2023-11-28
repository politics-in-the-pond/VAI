import mel
import slicer
import numpy as np
import io
import tqdm
import soundfile
import torchaudio
import content_extractor.hubert as hubert
from modelsrc.models3 import SynthesizerTrn
import utils
import torch
import librosa
from torch.utils.data import DataLoader
import copy
from dataloader_infer import AudioDataLoader, DataCollate
import os
import glob

### settings ###
slice_db = -40
speaker_id = 0
sample_rate = 44100
modelname = "haruka"

def vec_resampling(vec_src, dst_len):
    vec_len = vec_src.shape[-1]
    vec = torch.zeros([vec_src.shape[0], dst_len], dtype=torch.float)
    ratio = dst_len / vec_len
    position_checker = torch.arange(vec_len+1) * ratio
    position = 0
    for i in range(dst_len):
        if i >= position_checker[position+1]:
            position += 1
            vec[:, i] = vec_src[:, position]
        else:
            vec[:, i] = vec_src[:, position]
    return vec

def get_audio_vec(audiopath, model):
    audio, sr = librosa.load(audiopath, sr=sample_rate)
    audio = torch.FloatTensor(audio)
    audio = torchaudio.functional.resample(audio, sr, 16000)
    if len(audio.shape) == 2 and audio.shape[1] >= 2:
        audio = torch.mean(audio, dim=0).unsqueeze(0)
    audio = audio.unsqueeze(0).unsqueeze(0)
    with torch.inference_mode():
        units = model.units(audio)
        return units.transpose(1, 2).squeeze()

def get_CQT(audiopath):
    audio, sr = librosa.load(audiopath, sr=sample_rate)
    audio = torch.FloatTensor(audio)
    if len(audio.shape) == 2 and audio.shape[1] >= 2:
        audio = torch.mean(audio, dim=0).unsqueeze(0)
    audio = audio.numpy().astype('double')
    audio_norm = audio #/ hps.data.max_wav_value
    C = librosa.cqt(audio_norm, sr=sr)
    return torch.tensor(abs(C))

def get_mel_spec(audiopath):
    audio, sr = librosa.load(audiopath, sr=sample_rate)
    audio = torch.FloatTensor(audio)
    if sr != sample_rate:
        raise ValueError("{} SR doesn't match target {} SR".format(
            sr, sample_rate))
    if len(audio.shape) == 2 and audio.shape[1] >= 2:
        audio = torch.mean(audio, dim=0).unsqueeze(0)
    audio_norm = audio.unsqueeze(0)
    spec = mel.mel_spectrogram_torch(audio_norm, 1024, 80, sample_rate, 256, 1024, 0.0, None, center=False)
    return spec.squeeze()


def infer():
    torch.manual_seed(7650)
    path = r'./raw'
    result_path = r'./temp'
    files = os.listdir(path)
    pad_seconds = 0.5

    f = input("input file name. eg.) test.wav ")
    [os.remove(file) for file in glob.glob(r'./temp/*')]
    cnt = 0
    filepath = path + r'/' + f
    thresh = -20
    while True:
        chunks = slicer.cut(filepath, db_thresh=thresh)
        audio_data, audio_sr = slicer.chunks2audio(filepath, chunks)
        print(len(audio_data))
        if len(audio_data) > 20:
            break
        else:
            thresh += 2
    for (tag, data) in audio_data:
        print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')
        #if round(len(data) / audio_sr, 3) < 1:
        #    continue
        pad_len = int(audio_sr * pad_seconds)
        data = np.concatenate([np.zeros([pad_len]), data, np.zeros([pad_len])])
        soundfile.write(result_path + r'/' + str(cnt) + "_" + f, data, audio_sr, format="wav")
        cnt += 1

    filenames = os.listdir((r'./temp'))
    model = hubert.hubert_soft("./content_extractor/hubert-soft-0d54a1f4.pt")
    for i, f in enumerate(tqdm.tqdm(filenames)):
        origin_name = r'./temp/' + f[:-4]
        f = r'./temp/' + f
        audio, _ = torchaudio.load(f)
        if audio.shape[1] < 8192:
            print(f)
            continue

        spec = get_mel_spec(f)
        vec = get_audio_vec(f, model)
        cqt = get_CQT(f)

        np.save(origin_name + ".cqt", cqt.cpu().numpy())
        np.save(origin_name + ".vec", vec.cpu().numpy())
        np.save(origin_name + ".spec", spec.cpu().numpy())

    hps = utils.get_hparams()
    train_dataset = AudioDataLoader(f'.\\temp', hparams=hps.data)
    collate_fn = DataCollate()
    loader = DataLoader(train_dataset, num_workers=6, shuffle=False,
                              batch_size=1, collate_fn=collate_fn)
    net_g = SynthesizerTrn(
        80,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).cuda(0)
    net_g, _, _, _ = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g)
    net_g.eval()
    audio = []

    pad = int(44100*0.5) - 64

    for i, item in enumerate(tqdm.tqdm(loader)):
        vec, spec, cqt, lengths, wav_norm, wav_length, sid = item
        with torch.no_grad():
            out_audio = net_g.infer(vec.cuda(0), cqt.cuda(0), lengths.cuda(0), sid.cuda(0))
        _audio = out_audio.squeeze().squeeze().cpu().numpy()
        _audio = _audio[pad:-pad+1]
        audio.extend(list(_audio))
    filename = "_".join(f.split(r'/')[-1].split(r'_')[1:])
    res_path = f'./result/{filename}'
    soundfile.write(res_path, audio, sample_rate)

    [os.remove(file) for file in glob.glob(r'./temp/*')]

if __name__ == "__main__":
    infer()