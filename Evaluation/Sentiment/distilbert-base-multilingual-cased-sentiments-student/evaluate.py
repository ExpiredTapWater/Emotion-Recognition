#!/usr/bin/env python
# coding: utf-8

# ## Evaluate.ipynb
# #### **This notebook evaluates the accuracy of just one model on the IEMOCAP dataset ground truths.**
# 
# - **Sentiment Analysis Model:** lxyuan/distilbert-base-multilingual-cased-sentiments-student
# - Source: [HuggingFace](https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student)

# ## Setup

# In[ ]:


# Run
SEED = 22
FOLD = 0

# Models
SENTIMENT_MODEL_NAME = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"

# Flag to enable parsing of arguments when converted to script. Set true after converting
PYTHON_SCRIPT = True


# ### For Conversion to .py file

# In[ ]:


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


# ## Folders

# In[ ]:


# Filepaths
OUTPUT_FOLDER = f'./fold_{FOLD}'
LOG_OUTPUT = OUTPUT_FOLDER + f'/fold-{FOLD}.log'


# ### Dataset Setup

# In[ ]:


# Dataset
TSV = r'C:\Users\ChenYi\Downloads\AAI3001_Project\labels\IEMOCAP_4.tsv'
AUDIO_DIRECTORY = r'C:\Users\ChenYi\Downloads\AAI3001_Project\small-project\IEMOCAP_full_release_audio'

# Contains the correct transcription provided by the IEMOCAP dataset
GROUND_TRUTH = './groundtruth_IEMOCAP.csv'

# Contains the correct emotions (A, H, S, N) remapped to positive, negative and neutral
REMAPPED_EMOTIONS = './sentiment_truths_IEMOCAP.csv'


# #### Select GPUs (For multi-GPU setup)

# In[ ]:


import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ### Logger

# In[ ]:


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


# ### NLP Preprocessing Settings

# In[ ]:


APPLY_NLP_PREPROCESS = True
APPLY_CONTRACTIONS = True
APPLY_LEMMANTIZATION = True
APPLY_REMOVE_STOPWORDS = False
APPLY_REMOVE_NUMBERS = True  


# ### For Cloud Instances
# Run the following to install required libraries

# In[ ]:


CLOUD = False


# In[ ]:


if CLOUD:
    get_ipython().system('pip install contractions')
    get_ipython().system('pip install clean-text')
    get_ipython().system('pip install nltk')
    get_ipython().system('python -m spacy download en_core_web_sm')


# ## Imports

# In[ ]:


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
#from transformers import EarlyStoppingCallback, AdamW, get_scheduler
#from transformers import AutoTokenizer, AutoModelForSequenceClassification
#from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, pipeline
from transformers import pipeline
#from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import confusion_matrix, classification_report, recall_score, accuracy_score

# NLP Stuff
import spacy
import string
from nltk.corpus import stopwords
nlp = spacy.load("en_core_web_sm")
mapping = str.maketrans('', '', string.digits) # table to remove strings


# ### Log Details

# In[ ]:


logger.info("----- Models -----")
logger.info(f"Sentiment Model: {SENTIMENT_MODEL_NAME}")
logger.info("----- Parameters -----")
logger.info(f"Seed: {SEED}")
logger.info(f"Fold: {FOLD}")

logger.info("----- NLP Options -----")
if APPLY_NLP_PREPROCESS:
    logger.info(f"Apply Preprocessing: YES")
    logger.info(f"Remove Contractions: {APPLY_CONTRACTIONS}")
    logger.info(f"Apply Lemmantization: {APPLY_LEMMANTIZATION}")
    logger.info(f"Remove Numbers: {APPLY_REMOVE_NUMBERS}")
    logger.info(f"Remove Stopwords: {APPLY_REMOVE_STOPWORDS}") 
else:
    logger.info(f"Apply Preprocessing: NO")
logger.info("--------------------")


# ### Provided Code

# In[ ]:


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


# In[ ]:


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

# In[ ]:


# Load sentiment analysis model
#sentiment_task = pipeline("sentiment-analysis",
#                          model=SENTIMENT_MODEL_NAME,
#                          tokenizer=SENTIMENT_MODEL_NAME,
#                         device=0)

sentiment_task = pipeline(model=SENTIMENT_MODEL_NAME,tokenizer=SENTIMENT_MODEL_NAME,device=0)
#sentiment_task_full = pipeline(model=SENTIMENT_MODEL_NAME,tokenizer=SENTIMENT_MODEL_NAME,device=0,return_all_scores=True)

logger.info(F"Sentiment model loaded from {SENTIMENT_MODEL_NAME} successfully")


# ## Dataset & Loading

# In[ ]:


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
        #if mode == 'train':
        #    print("Each emotion has the following number of training samples:")
        #    print([[self.ClassNames[i], (self.label == i).sum()] for i in range(self.NumClasses)])
        self.weight = 1 / torch.tensor([(self.label == i).sum() for i in range(self.NumClasses)]).float()

    def get_classname(self):
        return self.ClassNames
    
    # Updated split_dataset function using fold
    
    def split_dataset(self, df_all, fold, speakers, mode):
        
        spk_len = len(speakers)
        #test_idx = np.array(df_all['speaker']==speakers[fold*2%spk_len])+np.array(df_all['speaker']==speakers[(fold*2+1)%spk_len])
        #val_idx = np.array(df_all['speaker']==speakers[(fold*2-2)%spk_len])+np.array(df_all['speaker']==speakers[(fold*2-1)%spk_len])
        #train_idx = True^(test_idx+val_idx)
        #train_idx = True^test_idx
        test_idx = np.array(df_all['speaker']==speakers[fold%spk_len])
        if fold%2==0:
            val_idx = np.array(df_all['speaker']==speakers[(fold+1)%spk_len])
        else:
            val_idx = np.array(df_all['speaker']==speakers[(fold-1)%spk_len])
        train_idx = True^(test_idx+val_idx)
        train_data_info = df_all[train_idx].reset_index(drop=True)
        val_data_info = df_all[val_idx].reset_index(drop=True)
        test_data_info = df_all[test_idx].reset_index(drop=True)
        #val_data_info = test_data_info = df_all[test_idx].reset_index(drop=True)
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


# In[ ]:


# Instantiate datasets
train_dataset = Mydataset(mode='train', max_len=6, fold=FOLD)
val_dataset = Mydataset(mode='val', max_len=6, fold=FOLD)
test_dataset = Mydataset(mode='test', max_len=6, fold=FOLD)

logger.info("Dataset Loaded")


# In[ ]:


# Put test information into a dataframe for later use
data_info = test_dataset.data_info
test_dataframe = data_info[['filename', 'label']].copy()
test_dataframe['filepath'] = test_dataframe['filename'].apply(
    lambda x: os.path.join(test_dataset.get_audio_dir_path, f"{x}.wav"))


# ### Run Model Prediction

# #### Clean and preprocess text for sentiment analysis
#     * Expanding contractions
#     * Removing punctuations
#     * Lemmatizing text
#     * Lowercasing
#     * Remove Numbers
#     * Removing stopwords

# In[ ]:


def NLP_Preprocess(string):
    
    # Check if function should run
    if APPLY_NLP_PREPROCESS:
     
        output = string

        if APPLY_CONTRACTIONS:
            
            # Expand Contractions
            words = string.split()
            output = [contractions.fix(word) for word in words]
            output = ' '.join(output)
            
        if APPLY_LEMMANTIZATION:

            doc = nlp(string)
            output = " ".join([token.lemma_ for token in doc])
            
        if APPLY_REMOVE_NUMBERS:

            # Remove Numbers
            output = output.translate(mapping)
            
        if APPLY_REMOVE_STOPWORDS:

            # Result show this reduces over accuracy
            doc = nlp(string)
            output = [token.text for token in doc if not token.is_stop]
            output= ' '.join(output)

        return output.lower()
    
    else:
        return string
    


# ## Obtain Predictions

# In[ ]:


def predict(test_dataframe):
    # Load ground truths
    ground_truths = pd.read_csv(GROUND_TRUTH)

    results = []
    total = test_dataframe.shape[0]
    count = 1

    # Iterate over each audio file in the test folder
    for index, row in test_dataframe.iterrows():
        # Display progress
        print(f'File {count} of {total}', end='\r')
        count += 1

        # Extract filename
        filename = row['filename']

        # Get transcription from ground_truths dataframe
        transcription = ground_truths.loc[ground_truths['ID'] == filename, 'Transcription'].values
        if transcription.size > 0:
            transcription = transcription[0]  # Extract transcription if found
            if transcription == " ":
                logger.info(f"Now on file: {filename}. Take note that this transcription is actually blank and not an error (i.e: ' ').")
            
        else:
            logger.info(f"Warning! Transciption not found for {filename}. Double check the dataset!")
            

        # Apply NLP preprocessing if required
        if APPLY_NLP_PREPROCESS:
            transcription_nlp = NLP_Preprocess(transcription)

        # Run sentiment analysis on the transcription
        sentiment = sentiment_task(transcription)
        sentiment_label = sentiment[0]['label']
        sentiment_score = sentiment[0]['score']
        
        # Debug
        print(f"Sentiment: {sentiment}")

        # Append the result to the list
        results.append([filename, transcription, transcription_nlp, sentiment_label, sentiment_score])

    logger.info(f"Done processing {count-1} files")

    # Write the results to a CSV file
    global CSV_FILEPATH
    CSV_FILEPATH = os.path.join(OUTPUT_FOLDER, "predictions.csv")

    with open(CSV_FILEPATH, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID','transcription','transcription_nlp','sentiment', 'sentiment_score'])
        writer.writerows(results)

    logger.info(f"Predictions saved to {CSV_FILEPATH}")


# In[ ]:


predict(test_dataframe)


# ## Evaluate Accuracy

# In[ ]:


# Function to perform accuracy evaluation
def calculate_accuracy(dataframe):
    logger.info("Now calculating accuracy")
    
    # Read the reference truth file
    reference = pd.read_csv(REMAPPED_EMOTIONS)

    # Merge DataFrames on 'filename'
    df_merged = pd.merge(
        reference[['filename', 'remapped_sentiment']],
        dataframe[['ID', 'sentiment']],
        left_on='filename',
        right_on='ID'
    )
    
    # dataframe contacts the files used in the test split (~500)
    # reference contains all files in the IEMOCAP dataset
    # df_merged will be a dataframe that contains only the specific files from that fold, and the corresponding truth from the reference

    # Extract true labels and predictions
    y_true = dataframe['sentiment']
    y_pred = df_merged['remapped_sentiment']

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create a DataFrame for the confusion matrix
    labels = sorted(y_true.unique())
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    
    # Do not show CM when running as a python script
    if not PYTHON_SCRIPT:
     
        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.ylabel('Actual Labels')
        plt.xlabel('Predicted Labels')
        plt.title('Confusion Matrix')
        plt.show()
    
    # Compute and print UA score
    macro_recall = recall_score(y_true, y_pred, average='macro')
    logger.info(f"Test UA: {macro_recall}")
    
    logger.info("Confusion Matrix:")
    logging.info(f"\n{cm_df}")

    # Generate classification report
    report = classification_report(y_true, y_pred, labels=labels)
    logger.info("Classification Report:")
    logging.info(f"\n{report}")


# ### Run Evalutation Function

# In[ ]:


df = pd.read_csv(CSV_FILEPATH)    
calculate_accuracy(df)


# In[ ]:




