#!/usr/bin/env python3

"""

Usage: 
The manifest directory is already created. Now it is necessary to create a combined dictionary. 


"""

from collections import Counter

char_counter = Counter()

for ltr_path in ["manifest/train.ltr", "manifest/valid.ltr"]:
    with open(ltr_path, "r", encoding="utf-8") as f:
        for line in f:
            chars = line.strip().split()
            for char in chars:
                # Keep single-character tokens that are either a letter, '|' or "'"
                if len(char) == 1 and (char.isalpha() or char in {"|", "'"}):
                    char_counter.update(char)

with open("manifest/dict.ltr.txt", "w", encoding="utf-8") as f:
    for char, count in sorted(char_counter.items()):
        f.write(f"{char} {count}\n")
      