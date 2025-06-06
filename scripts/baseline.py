# -*- coding: cp1252 -*-
#!/usr/bin/env python3
import os
import torchaudio
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from jiwer import wer
from tqdm import tqdm

# Paths
wav_dir = "/scratch/s4367073/thesis/eval_wavs"
reference_path = "/scratch/s4367073/thesis/evaluation.txt"
output_tsv = "/scratch/s4367073/thesis/results/wav2vec2_baseline_results.tsv"

# Load model and processor
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-dutch"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)
model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

# Load references (convert to lowercase here)
references = {}
with open(reference_path, "r", encoding="utf-8") as f:
    for line in f:
        fname, ref = line.strip().split("\t")
        references[fname] = ref.lower()  # force lowercase

# Process all wav files
results = []
device = "cuda" if torch.cuda.is_available() else "cpu"

for fname in tqdm(sorted(os.listdir(wav_dir))):
    if not fname.endswith(".wav"):
        continue

    wav_path = os.path.join(wav_dir, fname)

    # Load and preprocess audio
    speech_array, sr = torchaudio.load(wav_path)
    input_values = processor(
        speech_array.squeeze(),
        sampling_rate=sr,
        return_tensors="pt",
        padding=True
    ).input_values.to(device)

    # Predict
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0].lower()  # force lowercase

    # Get reference (already stored in lowercase)
    ref_text = references.get(fname, "")

    # Compute WER
    wer_score = wer(ref_text, transcription)

    # Save result
    results.append((fname, ref_text, transcription, round(wer_score, 4)))

# Compute average WER
if results:
    total_wer = sum(r[3] for r in results)
    avg_wer = total_wer / len(results)
    avg_wer = round(avg_wer, 4)
else:
    avg_wer = 0.0

# Write to .tsv (and include average at the bottom)
os.makedirs(os.path.dirname(output_tsv), exist_ok=True)
with open(output_tsv, "w", encoding="utf-8") as out_f:
    out_f.write("filename\tlabel\tprediction\twer\n")
    for row in results:
        out_f.write("\t".join(map(str, row)) + "\n")
    # Append an “AVERAGE” row at the end (optional)
    out_f.write(f"AVERAGE\t-\t-\t{avg_wer}\n")

print(f"\n? Done! Results saved to: {output_tsv}")
print(f"Average WER over {len(results)} files: {avg_wer}")
