#!/usr/bin/env python
# coding: utf-8

# ## Evaluate.ipynb
# #### **This notebook evaluates the accuracy of just one model on the IEMOCAP dataset ground truths.**
# 
# - **Transcription Model (S2T):** whisper-tiny.en
# - Source: [HuggingFace](https://huggingface.co/openai/whisper-tiny)

# ## Setup

# In[1]:


# Run
SEED = 22
FOLD = 0

# Models
TRANSCRIPTION_MODEL_NAME = "openai/whisper-tiny.en"

# Flag to enable parsing of arguments when converted to script. Set true after converting
PYTHON_SCRIPT = True


# ### For Conversion to .py file

# In[2]:


if PYTHON_SCRIPT:

    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--seed',default=2021,type=int)
    parser.add_argument('--fold',type=int,required=True)
    parser.add_argument('--remap',type=bool,required=True)
    parser.add_argument('--threshold',type=float,required=False)
    parser.add_argument('--mode',required=False)
    parser.add_argument('--flip',type=bool,required=False)

    # Parse the arguments
    args = parser.parse_args()

    # Run
    SEED = args.seed
    FOLD = args.fold
    RUN_REMAP = args.remap
    THRESHOLD = args.threshold
    MODE = args.mode
    FLIP = args.flip


# ### Folders

# In[3]:


# Filepaths
OUTPUT_FOLDER = f'./fold_{FOLD}'
LOG_OUTPUT = OUTPUT_FOLDER + f'/fold-{FOLD}.log'


# ### Dataset Setup

# In[4]:


# Dataset
TSV = r'C:\Users\ChenYi\Downloads\AAI3001_Project\labels\IEMOCAP_4.tsv'
AUDIO_DIRECTORY = r'C:\Users\ChenYi\Downloads\AAI3001_Project\small-project\IEMOCAP_full_release_audio'


# ### Select GPUs (For multi-GPU setup)

# In[5]:


import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ### Logger

# In[6]:


import logging
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)                                                                                                                                                                                     
    logger.addHandler(fh)                                                                                                                                                                                          
                                                                                                                                                                                                                   
    sh = logging.StreamHandler()                                                                                                                                                                                   
    sh.setFormatter(formatter)                                                                                                                                                                                     
    logger.addHandler(sh)                                                                                                                                                                                          
                                                                                                                                                                                                                   
    return logger
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
logger = get_logger(LOG_OUTPUT)


# ## Imports

# In[7]:


import csv
import torch
import random
import librosa
import torchaudio
import contractions
import torch.nn as nn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict
from torchaudio import functional as audioF
from torchaudio.transforms import Resample
from torchaudio.compliance import kaldi
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, recall_score, accuracy_score
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline


# ### Log Details

# In[8]:


logger.info("----- Models -----")
logger.info(f"Speech-To-Text (Transcription) Model: {TRANSCRIPTION_MODEL_NAME}")
logger.info("----- Parameters -----")
logger.info(f"Seed: {SEED}")
logger.info(f"Fold: {FOLD}")
logger.info("--------------------")


# ### Provided Code

# In[9]:


class Pad_trunc_wav(nn.Module):
    def __init__(self, max_len: int = 6*16000):
        super(Pad_trunc_wav, self).__init__()
        self.max_len = max_len
    def forward(self,x):
        shape = x.shape
        length = shape[1]
        if length < self.max_len:
            multiple = self.max_len//length+1
            x_tmp = torch.cat((x,)*multiple, axis=1)
            x_new = x_tmp[:,0:self.max_len]
        else:
            x_new = x[:,0:self.max_len]
        return x_new


# In[10]:


def setup_seed(seed=2021):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
setup_seed(SEED)


# ### Download Required Models

# In[11]:


# Load whisper model and processor for speech-to-text
S2T_processor = WhisperProcessor.from_pretrained(TRANSCRIPTION_MODEL_NAME)
S2T_Model = WhisperForConditionalGeneration.from_pretrained(TRANSCRIPTION_MODEL_NAME)
S2T_Model = S2T_Model.to('cuda')
feature_extractor=S2T_processor.feature_extractor

logger.info(F"Speech-To-Text (Transcription) model loaded from {TRANSCRIPTION_MODEL_NAME} successfully")


# ### Dataset & Loading

# In[12]:


class Mydataset(Dataset):
    def __init__(self, mode='train', max_len=6, seed=2021, fold=0, data_path=TSV, audio_dir=AUDIO_DIRECTORY):
        self.mode = mode
        data_all = pd.read_csv(data_path, sep='\t')
        SpkNames = np.unique(data_all['speaker'])  # ['Ses01F', 'Ses01M', ..., 'Ses05M']
        self.data_info = self.split_dataset(data_all, fold, SpkNames, mode)
        self.get_audio_dir_path = os.path.join(audio_dir)
        self.pad_trunc = Pad_trunc_wav(max_len * 16000)
         
        # Label encoding
        self.label = self.data_info['label'].astype('category').cat.codes.values
        self.ClassNames = np.unique(self.data_info['label'])
        self.NumClasses = len(self.ClassNames)
        self.weight = 1 / torch.tensor([(self.label == i).sum() for i in range(self.NumClasses)]).float()

    def get_classname(self):
        return self.ClassNames
    
    # Updated split_dataset function using fold
    
    def split_dataset(self, df_all, fold, speakers, mode):
        
        spk_len = len(speakers)
        test_idx = np.array(df_all['speaker']==speakers[fold%spk_len])
        if fold%2==0:
            val_idx = np.array(df_all['speaker']==speakers[(fold+1)%spk_len])
        else:
            val_idx = np.array(df_all['speaker']==speakers[(fold-1)%spk_len])
        train_idx = True^(test_idx+val_idx)
        train_data_info = df_all[train_idx].reset_index(drop=True)
        val_data_info = df_all[val_idx].reset_index(drop=True)
        test_data_info = df_all[test_idx].reset_index(drop=True)

        if self.mode == 'train':
            data_info = train_data_info
        elif self.mode == 'val':
            data_info = val_data_info
        elif self.mode == 'test':
            data_info = test_data_info
        else:
            data_info = df_all
        
        logger.info(f"Mode: {mode} Fold: {fold}")
        return data_info

    def pre_process(self, wav):
        
        if self.mode == 'test': 
            return wav
        else:
            wav = self.pad_trunc(wav)
            return wav

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        # Load the raw waveform from file using data_info to get filenames
        wav_path = os.path.join(self.get_audio_dir_path, self.data_info['filename'][idx]) + '.wav'
        wav, sample_rate = torchaudio.load(wav_path)

        # Preprocess the waveform (e.g., pad/truncate if needed)
        wav = self.pre_process(wav)

        # Apply Wav2Vec2 feature extractor
        inputs = feature_extractor(
            wav.squeeze().numpy(),  # Convert PyTorch tensor to numpy array
            sampling_rate=sample_rate,
            return_tensors="pt",  # Return PyTorch tensors
            padding=True  # Optionally pad to a fixed length
        )

        label = self.label[idx]

        # Return the processed input values and the label
        return {
            'input_values': inputs['input_values'].squeeze(0),  # Remove extra batch dimension
            'labels': torch.tensor(label, dtype=torch.long)}


# In[13]:


# Instantiate datasets
test_dataset = Mydataset(mode='test', max_len=6, fold=FOLD)

logger.info("Dataset Loaded")


# In[14]:


# Put test information into a dataframe for later use
data_info = test_dataset.data_info
test_dataframe = data_info[['filename', 'label']].copy()
test_dataframe['filepath'] = test_dataframe['filename'].apply(
    lambda x: os.path.join(test_dataset.get_audio_dir_path, f"{x}.wav"))


# ## Perform Inference and Obtain Transcriptions

# #### Function to clean up transcription to match what is required for compute_wer.py
# - Remove full stop
# - Remove all punctuations except apostrophe
# - Change all to UPPERCASE

# In[15]:


def format_transcription(ID, text):
    
    original_text = text  # Store the original text for comparison
    
    # Remove full stop
    text = text.replace('.', '')
    
    # Remove all punctuations except apostrophe
    text = ''.join(char if char.isalnum() or char == "'" else ' ' for char in text)
    
    # Convert all text to uppercase
    text = text.upper()

    # Alert if any formatting was done:
    if text != original_text:
        logger.info(f"Transcription of file {ID} has been formatted")
        logger.info(f"ORIGINAL: {original_text}")
        logger.info(f"FORMATTED: {text}")
    
    return text


# In[16]:


def predict(test_dataframe):

    results = []
    total = test_dataframe.shape[0]
    count = 1

    # Iterate over each audio file in the test folder
    for index, row in test_dataframe.iterrows():

        # Display progress
        print(f'File {count} of {total}', end='\r')
        count += 1

        # Load audio file
        filename = row['filename'] + '.wav'
        audio_file = os.path.join(AUDIO_DIRECTORY, filename)
        audio, sample_rate = librosa.load(audio_file, sr = 16000)
        
        # Tokenize the input audio for speech-to-text model
        input_features = S2T_processor(audio, return_tensors="pt", sampling_rate=16000, padding="longest").input_features
        input_features = input_features.to('cuda')
        predicted_ids = S2T_Model.generate(input_features)
        
        transcription = S2T_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        # Important!
        # Batch size should be 1 for inference. I am not using a dataloader, and iterating per file, so my batch size is effectively one
        # Padding="longest" pads each file to the longest in each batch, but because my batch is 1, effectively no padding is done.
        # For clarify, padding="do_not_pad" is used
        
        # Obtain transcriptions
        #with torch.no_grad():
        #    S2T_logits = S2T_Model(input_values).logits
        #    predicted_ids = torch.argmax(S2T_logits, dim=-1)
        #    transcription = S2T_processor.batch_decode(predicted_ids)[0]
            
        # Perform cleaning
        formatted_transcription = format_transcription(count, transcription)

        # Extract the filename without the extension
        filename = os.path.splitext(os.path.basename(audio_file))[0]

        # Append the result to the list
        results.append([filename,
                        formatted_transcription])

    logger.info(f"Done processing {total} files")

    # Write the results to a CSV file
    global CSV_FILEPATH
    CSV_FILEPATH = os.path.join(OUTPUT_FOLDER, "transcriptions.csv")

    with open(CSV_FILEPATH, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID', 'transcription'])
        writer.writerows(results)

    logger.info(f"Transcriptions saved to {CSV_FILEPATH}")
    return CSV_FILEPATH


# In[ ]:


predict(test_dataframe)


# ## Convert CSV into TXT file
# - Optional step to convert into TXT file for use in compute_wer.py
# - This function can also be found in notebooks/TranscriptionFormatter.ipynb, where you can optionaly convert each fold at once

# In[18]:


# Input and output file paths
input_csv = CSV_FILEPATH  # Replace with your CSV file path
output_txt = OUTPUT_FOLDER + "/transcriptions.txt"  # Replace with your desired TXT file path


# In[19]:


# Open the CSV file and read its content
def convert_csv_to_txt(input_csv, output_txt):
    with open(input_csv, mode="r", encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader, None)  # Skip header if present
        with open(output_txt, mode="w", encoding="utf-8") as txt_file:
            for row in csv_reader:
                # Combine ID and Transcript with a space separator
                txt_file.write(f"{row[0]} {row[1]}\n")
    
convert_csv_to_txt(input_csv, output_txt)
print(f"Converted and formatted {input_csv} to {output_txt} for use in computing word error rate.")


# In[ ]:




