# VAI
VITS2 based Audio Inference model (Voice Conversion)
<img width="991" alt="스크린샷 2024-03-07 202220" src="https://github.com/politics-in-the-pond/VAI/assets/74289147/8004f304-e798-4309-b29f-db481d0fc37d">

## Before start...
Get [HuBERT model](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt) and put on '/content_extractor/'<br><br>
Collect 44100Hz audios and put on '/train_wavs/{your_model_name}/'<br><br>
Run 'preprocessing.py'<br><br>
Copy some files from '/train_wavs/{your_model_name}/' to '/val_wavs/{your_model_name}/' (to validate model)<br><br>
Run 'train.py'<br>

## Train

## Inference
Get [slicer](https://github.com/prophesier/diff-svc/blob/main/infer_tools/slicer.py) and put on the root of code.

## Examples

## References
[VITS2 code](https://github.com/p0p4k/vits2_pytorch)

