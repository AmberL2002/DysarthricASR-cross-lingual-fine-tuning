#!/usr/bin/env python3
import torch
import torchaudio
import fairseq
import argparse
import glob
import os
import librosa
import numpy as np
from itertools import groupby
import re
from pyctcdecode import build_ctcdecoder
from jiwer import wer as jiwer_wer
   
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_cp", required=True)
    parser.add_argument("--wav_dir", required=True)
    parser.add_argument("--path_to_dict", required=True)
    parser.add_argument("--path_to_trans", required=True)
    parser.add_argument("--lm", default=None)
    parser.add_argument("--beam_width", default=None, type=int, required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--out_name", required=True)
    args = parser.parse_args()

    print(f"Arguments: {args}")

    # Define the model
    print("Defining the model")
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([args.path_to_cp], arg_overrides={"data": "/scratch/s4367073/thesis/manifest/"})
    model = model[0]
    model.eval()
    
    # Make the token lists 
    print("Making the token lists")
    tokens_lst = ["<pad>", "<s>", "</s>", "<unk>"]
    with open(args.path_to_dict) as f:
        for line in f.readlines():
            tokens_lst.append(line[0])

    
    # Build CTC decoder
    print("Building the CTC decoder")
    decoder = build_ctcdecoder(labels=tokens_lst, kenlm_model_path=args.lm)

    # Obtain data samples as a list
    print("Obtaining data samples as a list")
    data = glob.glob(os.path.join(args.wav_dir, "*.wav"), recursive=True)
    print(f"Data: {data}")
    

    # Make a look-up table from audio to transcript
    print("Making a look-up table from audio to transcript")
    audio_to_trans = {}
    with open(args.path_to_trans) as f:
        for line in f.readlines():
            audio_to_trans[line.split("\t")[0]] = line.split("\t")[1].rstrip("\n")
    
    print(audio_to_trans)
    # Evaluation
    print("Starting evaluation")
    with torch.no_grad():
        wer = 0    
        preds = []
        true = []
        
        results = []

        for sample in data:
            print(sample)
            audio_name = sample.split("/")[-1]
            actual_transcript = ""
            print(f"Evaluating {audio_name}")
            if audio_name in audio_to_trans:
                actual_transcript = audio_to_trans[audio_name]
            elif sample in audio_to_trans:
                actual_transcript = audio_to_trans[sample]
            actual_transcript = actual_transcript.lower()
            waveform, sr = torchaudio.load(sample)
            emission = model(source=waveform, padding_mask=None)
            logits = torch.squeeze(emission["encoder_out"], 1)
            
            beam_trans = decoder.decode(logits.detach().numpy(), beam_width=args.beam_width).lower()
            beam_wer = jiwer_wer(actual_transcript, beam_trans)
            
            wer += beam_wer
            preds.append(beam_trans)
            true.append(actual_transcript)
            
            results.append([audio_name + '\t', f'Label: {actual_transcript}' + '\t', f'Pred: {beam_trans}' + '\t', f'WER: {str(beam_wer)}'])

        wer = str(wer / len(true))
        print(f"WER:{wer}")

        results.append([f"Total WER: {wer}"])

        name = args.out_name + "_" + str(args.beam_width) + ".txt"

        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
            
        out_file = os.path.join(args.out_dir, name)

        with open(out_file, 'w') as f:
            f.write("\n".join(" ".join(line) for line in results))

if __name__ == '__main__':
    main()


