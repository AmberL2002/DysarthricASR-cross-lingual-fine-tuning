# Dysarthric ASR cross-lingual fine-tuning
For my Master's Thesis, I conducted a cross-lingual experiment to improve dysarthric ASR. Therefore I used for fine-tuning typical Dutch speech from Common Voice 13.0 delta. For fine-tuning with English dyasarthric speech, I used TORGO. For testing, I used the dataset COPAS together with Domotica, where only for COPAS it is necessary to ask for a license. 
The thesis can be downloaded from [here](https://campus-fryslan.studenttheses.ub.rug.nl/356/). #needs the correct link when thesis has been uploaded!!

### Table of Contents
1. [Available Features](#feature)
2. [Installation](#installation)
3. [Dataset Preparation](#dataset_preparation)
4. [Fine-tuning](#finetuning)
5. [Inference](#inference)
6. [Logs](#logs)

<a name = "feature" ></a>
### Available features
- SSH connection
- Multi-GPU training

<a name = "installation" ></a>
### Installation
- Make sure to use a Virtual environment.
- module load Python/3.10.8-GCCcore-12.2.0
- Download the XLSR-53 checkpoint from [this](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec#wav2vec-20)  website. 
```
cd fairseq
pip install --editable ./
pip install tensorboardX
pip install soundfile
pip install editdistance
```
- Make sure to change back to the directory with requirements.txt
```
pip install -r requirements.txt
```
- For the baseline, use [this](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-dutch) Huggingface model to make sure to model runs. The script is added to this repo, with the name [baseline.py](scripts/baseline.py). 

<a name = "dataset_preparation" ></a>
### Dataset Preparation
- Datasets need to be in **.txt** format and the audio files need to be in **.wav** format. 
- The transcription files need to be tab separated with the first column of each row the name of the audio file and the transcription of the wav. The files need to be samples at 16 kHz.
- In [create_manifest_from_txt.py](scripts/create_manifest_from_txt.py) .tsv files are created in a manifest directory with the amount of tokens in the sound file. It creates train.tsv and valid.tsv. Make sure to create separate english and dutch train.tsv files.
- Then, create parallel audio and label files. To generate label files for the fine-tuning and validation dataset run the following 2 commands with this [generate_labels.py](scripts/generate_labels.py) script.
```
python scripts/generate_labels.py \\ 
--transcriptions-file /PATH/TO/TRAIN_TSV \\
--output-dir /PATH/TO/MANIFEST_DIRECTORY \\
--output-name train

python scripts/generate_labels.py \\ 
--transcriptions-file /PATH/TO/VALID_TSV \\
--output-dir /PATH/TO/MANIFEST_DIRECTORY \\
--output-name valid
```
- Create a dict file with this [create_dict.py](scripts/create_dict.py) python script. Make sure to create 2 separate .dict files for both English and Dutch. Copy this dict into the manifest directory.

<a name = "finetuning" ></a>
### Fine-tuning
- Use the .yaml script from this repo. This can be found in the [config](config) folder. 
- Run the following command:
```
fairseq-hydra-train \\
task.data = /PATH/TO/MANIFEST_DIRECTORY \\
model.w2v_path = /PATH/TO/PRETRAINED_MODEL_TO_FINETUNE
checkpoint.restore_file = /PATH/TO/CHECKPOINT_TO_START_FINETUNING
distributed_training.distributed_world_size=1 \\
optimization.update_freq='[1]' \\
dataset.valid_subset = valid \\
--config-dir config \\
--config-name finetune_config.yaml
``` 

<a name = "inference" ></a>
### Inference
- For testing the model with the testing dataset, execute the command below using the [inference_beam_search.py](scripts/inference_beam_search.py)
- Make sure that this needs to be done two times, 1 for English typical dataset and Dutch dysarthric dataset. Also, keep in mind naming the files after the language, which makes them distinctive. 
```
python scripts/inference_beam_search.py \\
	--path_to_cp /PATH/TO/FINE-TUNED/MODELS/CHECKPOINT \\ 
	--wav_dir /PATH/TO/EVALUATION_WAVS_FOLDER \\
	--path_to_trans /PATH/TO/TRANSCRIPTION/FILE \\
	--path_to_dict dictionary/dict.ltr.txt \\
	--out_dir /PATH/TO/OUTPUT/DIRECTORY/TO/SAVE/RESULTS \\
	--out_name NAME_OF_THE_OUTPUT_FILE \\
	--beam_width 50
```
<a name = "logs" ></a>
### Logs
I ran Logs in Google Colab, since that worked better for me. 
```
!pip install tensorboardcolab
```

```
from google.colab import files
uploaded = files.upload()
```
Run the script for every log file seperately. 
```
import os
import shutil

os.makedirs("logs/run1", exist_ok=True)
shutil.move("/content/combined.out.tfevents.1749479237.a100gpu6", "/content/logs/run1")
```

```
%load_ext tensorboard
%tensorboard --logdir logs/
```
