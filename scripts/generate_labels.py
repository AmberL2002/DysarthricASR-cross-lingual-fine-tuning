#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""

Usage:
    python3 generate_labels.py \
        --transcriptions-file evaluation.txt \
        --output-dir ./labels \
        --output-name evaluation_wavs

This will create:
    ./labels/evaluation_wavs.ltr
    ./labels/evaluation_wavs.wrd


"""

import argparse
import os
import csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcriptions-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.transcriptions_file, "r") as tsv, open(
            os.path.join(args.output_dir, args.output_name + ".ltr"), "w"
    ) as ltr_out, open(
        os.path.join(args.output_dir, args.output_name + ".wrd"), "w"
    ) as wrd_out:

        # get a dict between the wavs and their transcriptions
        train_dir = next(tsv).strip()
        trans_path = train_dir+"/"+train_dir.split(os.path.sep)[-1]+".trans.txt"

        wav_file_transcription_dict = {}
        with open(trans_path, "r") as trans_f:
            for tline in trans_f:
                wav_name = tline.split("\t")[0].split("/")[-1]
                transcription = tline.split("\t")[1].strip().upper()
                transcription = transcription.replace(",","").replace(".","")
                wav_file_transcription_dict[wav_name] = transcription

        for line in tsv:
            wav_name = line.strip().split("\t")[0]
            transcription = wav_file_transcription_dict.get(wav_name)
            if transcription and transcription.strip() != "":
                # Each letter followed by a space, spaces become '| '
                # This guarantees every letter (including the last one) is followed by a space before the '|'
                ltr_tokens = []
                for char in transcription:
                    if char == " ":
                        ltr_tokens.append("|")
                    else:
                        ltr_tokens.append(char)
                # Add space after every token, including the last one, then join and strip
                transcription_ltr = " ".join(ltr_tokens) + " |"
                ltr_out.write(transcription_ltr.strip() + "\n")
                wrd_out.write(transcription.strip() + "\n")

if __name__ == "__main__":
    main()
