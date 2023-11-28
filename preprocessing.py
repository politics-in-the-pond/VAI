import pyworld
import torchaudio
import librosa
import numpy as np
import torch
import mel
import utils
import glob
import tqdm
from configs import audio_config
from content_extractor import hubert
from matplotlib import pyplot as plt

def get_f0_info(audiopath):
    audio, sr = torchaudio.load(audiopath)
    if len(audio.shape) == 2 and audio.shape[1] >= 2:
        audio = torch.mean(audio, dim=0).unsqueeze(0)
    audio = audio.numpy().squeeze(0).astype('double')
    f0, t = pyworld.dio(audio, sr)
    f0 = pyworld.stonemask(audio, f0, t, sr)
    plt.plot(range(0, len(f0)), f0)
    plt.show()

def get_CQT(audiopath, plot=False):
    audio, sr = torchaudio.load(audiopath)
    if len(audio.shape) == 2 and audio.shape[1] >= 2:
        audio = torch.mean(audio, dim=0).unsqueeze(0)
    audio = audio.numpy().squeeze(0).astype('double')
    audio_norm = audio #/ hps.data.max_wav_value
    C = librosa.cqt(audio_norm, sr=sr)
    if plot:
        C = np.abs(C)
        fig, ax = plt.subplots()
        img = librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max), sr=sr, x_axis='time', y_axis='cqt_note', ax=ax)
        ax.set_title('Constant-Q power spectrum')
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        plt.show()
    return torch.tensor(abs(C)).squeeze()

def get_mel_spec(audiopath, plot=False):
    audio, sr = torchaudio.load(audiopath)
    if sr != audio_config.sampling_rate:
        raise ValueError("{} SR doesn't match target {} SR".format(
            sr, audio_config.sampling_rate))
    if len(audio.shape) == 2 and audio.shape[1] >= 2:
        audio = torch.mean(audio, dim=0).unsqueeze(0)
    audio_norm = audio
    spec = mel.mel_spectrogram_torch(audio_norm, audio_config.filter_length, audio_config.n_mel_channels, audio_config.sampling_rate, audio_config.hop_length, audio_config.win_length, audio_config.mel_fmin, audio_config.mel_fmax, center=False)
    if plot:
        utils.plot_spectrogram_to_numpy(spec[0].data.numpy())
    return spec.squeeze()

def get_audio_vec(audiopath, model):
    audio, sr = torchaudio.load(audiopath)
    audio = torchaudio.functional.resample(audio, sr, 16000)
    if len(audio.shape) == 2 and audio.shape[1] >= 2:
        audio = torch.mean(audio, dim=0).unsqueeze(0)
    audio = audio.unsqueeze(0)
    with torch.inference_mode():
        units = model.units(audio)
        return units.transpose(1, 2).squeeze()

def match_hub_to_cqt(hub, cqt_shape):
    hub = torch.nn.functional.interpolate(hub, size=cqt_shape[1], scale_factor=None, mode='linear', align_corners=None, recompute_scale_factor=None)
    return hub

def get_energy(spec):
    spec = torch.mean(spec, dim=0, keepdim=True)
    return spec.squeeze()

def main():
    filenames = glob.glob(f'./train_wavs/*/*.wav', recursive=True)
    model = hubert.hubert_soft("./content_extractor/hubert-soft-0d54a1f4.pt")
    for i, f in enumerate(tqdm.tqdm(filenames)):
        origin_name = f[:-4]

        spec = get_mel_spec(f)
        vec = get_audio_vec(f, model)
        cqt = get_CQT(f, plot=True)
        # energy = get_energy(spec)
        # get_f0_info(f)

        np.save(origin_name + ".cqt", cqt.cpu().numpy())
        np.save(origin_name + ".vec", vec.cpu().numpy())
        np.save(origin_name + ".spec", spec.cpu().numpy())
        # np.save(origin_name + ".energy", energy.cpu().numpy())

main()
