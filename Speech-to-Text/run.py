import torch
import torch.nn.functional as F
import torchaudio
from torch.nn import CTCLoss

from models.baseline import Baseline
import processors.feature_extraction  as utils



model_name = "facebook/s2t-small-librispeech-asr"
processor = utils.load_model_and_processor(model_name)

# Path to your audio file
audio_path = "file_example_WAV_1MG.wav"

# Transcribe the audio file
waveform, sr = utils.load_audio(audio_path)
waveform = utils.normalize_audio(waveform)
waveform = utils.trim_silence(waveform, sr)

inputs = utils.prepare_inputs(waveform, processor, sr)
input_features = inputs.input_features

# Example to tokenize
string = "Hola, molta sort amb el Projecte"
tokens = (utils.tokenize_transcript(string, processor))

model = Baseline(input_dim=inputs.input_features.shape[1], hidden_dim=128, num_classes=processor.tokenizer.vocab_size)


outputs = model(input_features, tokens)

print(outputs.shape)
print("EXECUTION DONE")