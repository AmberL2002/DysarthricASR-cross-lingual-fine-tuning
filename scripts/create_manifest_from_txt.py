#!/usr/bin/env python3
import os
import soundfile

def create_tsv(txt_file, audio_dir, out_tsv):
    audio_dir = os.path.abspath(audio_dir)
    with open(txt_file, "r", encoding="utf-8") as infile, open(out_tsv, "w", encoding="utf-8") as out:
        # First line is the audio directory (absolute path)
        print(audio_dir, file=out)
        for line in infile:
            if '\t' not in line:
                continue
            audio_file = line.split('\t', 1)[0].strip()
            wav_path = os.path.join(audio_dir, audio_file)
            if not os.path.exists(wav_path):
                print(f"Warning: {wav_path} does not exist, skipping.")
                continue
            frames = soundfile.info(wav_path).frames
            print(f"{audio_file}\t{frames}", file=out)

if __name__ == "__main__":
    os.makedirs("manifest", exist_ok=True)
    # Create valid.tsv from evaluation.txt and evaluation_wavs/
    #create_tsv("evaluation.txt", "evaluation_wavs", "manifest/valid.tsv")
    # Create train.tsv from train_nl.txt and train_nl_wavs/
    create_tsv("eval_wavs/eval_wavs.trans.txt", "eval_wavs", "manifest/valid.tsv")
