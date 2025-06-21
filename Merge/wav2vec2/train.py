# Import
import os
import csv
import torch
import random
import logging
import librosa
import torchaudio
import contractions
import torch.nn as nn
import pandas as pd
import numpy as np
import transformers
from datasets import Dataset, DatasetDict
from torchaudio import functional as audioF
from torchaudio.transforms import Resample
from torchaudio.compliance import kaldi
from torch.utils.data import Dataset, DataLoader
from transformers import EarlyStoppingCallback, AdamW, get_scheduler
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, pipeline
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import confusion_matrix, classification_report, recall_score, accuracy_score

# Hardcoded filepaths
TSV = r'C:\Users\ChenYi\Documents\Github\SIT-emotion-recognition\baselines\data\meta\IEMOCAP_4.tsv'
AUDIO_DIRECTORY = r'C:\Users\ChenYi\Documents\Github\SIT-emotion-recognition\baselines\data\IEMOCAP_full_release_audio'

# .ipynb to .py conversion
import argparse

parser = argparse.ArgumentParser(description="wav2vec2-base")
parser.add_argument('--epochs',default=100,type=int)
parser.add_argument('--earlystopping',default=10,type=int)
parser.add_argument('--batch-size',default=16,type=int)
parser.add_argument('--seed',default=2021,type=int)
parser.add_argument('--lr',default=1e-3,type=float)
parser.add_argument('--fold',type=int,required=True)

# Parse the arguments
args = parser.parse_args()

# Access variables
epochs = args.epochs
earlystopping = args.earlystopping
batch_size = args.batch_size
seed = args.seed
learning_rate = args.lr
fold = args.fold

# Use arguments
EPOCH = epochs
LEARNING_RATE = learning_rate
EARLY_STOPPING = earlystopping
NUM_WORKERS = 0 # Use 0 For local Jupyter
BATCH_SIZE = batch_size  # Max 8 To fit within 10GB GPU memory

# Output Filepaths
OUTPUT_FILEPATH = f"./fold_{fold}" + "/wav2vec2-test"
LOGS = f"./fold_{fold}/fold{fold}_logs.log"

# Create directory
os.makedirs(f"./fold_{fold}", exist_ok=True)

# Logging Function
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

logger = get_logger(LOGS)

# Provided code (from small-project)
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
    
# Seed Function
def setup_seed(seed=2021):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
setup_seed(seed)

# Logging Callback Function
class LogCallback(transformers.TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        if len(state.log_history) == 0:
            logger.info(f"No log history available on first callback.")
            return

        logger.info(f"Epoch {state.epoch-1} finished.")
        last_log = state.log_history[-1]

        if "eval_loss" in last_log:
            logger.info(f"Evaluation Loss: {last_log['eval_loss']}")
        if "eval_accuracy" in last_log:
            logger.info(f"Evaluation Accuracy: {last_log['eval_accuracy']}")
           
    # Only the 2nd epoch will have logs, so everything is delayed by 1 epoch. Add the last epoch's logs manually
    def on_train_end(self, args, state, control, **kwargs):
        
        log_entry = state.log_history[-2]
        logger.info(f"Epoch {log_entry['epoch']} finished. (Last Epoch)")
        logger.info(f"Evaluation Loss: {log_entry['eval_loss']}")
        logger.info(f"Evaluation Accuracy: {log_entry['eval_accuracy']}")
        logger.info(f"----- Training Completed -----")
        
# Load base model from Hugging Face
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels = 4)

# Dataset Class (Using new split_dataset for folds)
class Mydataset(Dataset):
    def __init__(self, mode='train', max_len=6, seed=42, fold=0, data_path=TSV, audio_dir=AUDIO_DIRECTORY):
        self.mode = mode
        data_all = pd.read_csv(data_path, sep='\t')
        SpkNames = np.unique(data_all['speaker'])  # ['Ses01F', 'Ses01M', ..., 'Ses05M']
        self.data_info = self.split_dataset(data_all, fold, SpkNames)
        self.get_audio_dir_path = os.path.join(audio_dir)
        self.pad_trunc = Pad_trunc_wav(max_len * 16000)
         
        # Label encoding
        self.label = self.data_info['label'].astype('category').cat.codes.values
        self.ClassNames = np.unique(self.data_info['label'])
        self.NumClasses = len(self.ClassNames)
        if mode == 'train':
            print("Each emotion has the following number of training samples:")
            print([[self.ClassNames[i], (self.label == i).sum()] for i in range(self.NumClasses)])
        self.weight = 1 / torch.tensor([(self.label == i).sum() for i in range(self.NumClasses)]).float()

    def get_classname(self):
        return self.ClassNames
    
    # Updated split_dataset function using fold
    
    def split_dataset(self, df_all, fold, speakers):
        
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
        
        return data_info

    def pre_process(self, wav):
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

# Model Training Setup 
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    accuracy = np.sum(preds == labels) / len(labels)
    return {"accuracy": accuracy}

# Function to test model
def test_model(test_dataframe, model=''):
    
    logger.info("Testing on test dataset, please wait...")
    
    results = []
    total = test_dataframe.shape[0]
    count = 1
    
    logger.info(f"Total files: {total}")
    
    # Run predictions on test dataset
    for index, row in test_dataframe.iterrows():

        count += 1

        # Load audio file
        filename = row['filename'] + '.wav'
        audio_file = os.path.join(AUDIO_DIRECTORY, filename)
        y_ini, sr_ini = librosa.load(audio_file, sr = 16000)

        inputs = feature_extractor(y_ini, sampling_rate=16000, return_tensors="pt")
        
        # Send to GPU
        inputs.to('cuda')

        # Get the logits from the model
        with torch.no_grad():
            logits = model(**inputs).logits

        # Predict the class with the highest logit value
        predicted_class_id = torch.argmax(logits).item()

        # Append the result to the list
        results.append([row['filename'], predicted_class_id])

    # Format to dataframe
    prediction_dataframe = pd.DataFrame(results, columns=['ID', 'Predict'])


    # Load true values
    true_dataframe = pd.read_csv(TSV, sep='\t')
    remap_dict = {
        0: 'A',
        1: 'H',
        2: 'N',
        3: 'S'}

    # Remap predicted values to match TSV
    prediction_dataframe['Predict'] = prediction_dataframe['Predict'].map(remap_dict)

    # Merge DataFrames on 'filename'
    df_merged = pd.merge(true_dataframe[['filename', 'label']],prediction_dataframe[['ID', 'Predict']],
                         left_on='filename',right_on='ID')

    # Extract true labels and predictions
    y_true = df_merged['label']
    y_pred = df_merged['Predict']
    
    # Compute and print UA score
    macro_recall = recall_score(y_true, y_pred, average='macro')
    logger.info(f"----- Test Results -----")
    logger.info(f"Test UA: {macro_recall}")
        
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create a DataFrame for the confusion matrix
    labels = sorted(y_true.unique())
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    # Compute and print classification report
    report = classification_report(y_true, y_pred, labels=labels)    
    logger.info("Classification Report:")
    logger.info("\n%s", report)

def main():

    # Show arguments
    logger.info("Training with the following parameters:")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Early Stopping Patience: {earlystopping}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Random Seed: {seed}")
    logger.info(f"Learning Rate: {learning_rate}")
    logger.info(f"Fold: {fold}")
    
    # Instantiate datasets
    train_dataset = Mydataset(mode='train', max_len=6, seed=seed, fold=fold)
    val_dataset = Mydataset(mode='val', max_len=6, seed=seed, fold=fold)
    test_dataset = Mydataset(mode='test', max_len=6, seed=seed, fold=fold)

    # Put test information into a dataframe
    data_info = test_dataset.data_info
    test_dataframe = data_info[['filename', 'label']].copy()
    test_dataframe['filepath'] = test_dataframe['filename'].apply(
        lambda x: os.path.join(test_dataset.get_audio_dir_path, f"{x}.wav"))
    
    # Define the early stopping callback
    early_stopping = EarlyStoppingCallback(early_stopping_patience = EARLY_STOPPING)

    training_args = TrainingArguments(
        output_dir= f"./fold_{fold}",
        logging_dir=f"./fold_{fold}",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size = BATCH_SIZE,
        num_train_epochs = EPOCH,
        save_steps = 10,
        save_total_limit = 2,
        fp16 = True,
        dataloader_pin_memory = True,
        load_best_model_at_end = True,
        dataloader_num_workers = NUM_WORKERS,
        report_to = "none",
        gradient_checkpointing = True,
        learning_rate = LEARNING_RATE
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping, LogCallback]
        )
        
    # Start Training
    logger.info('----- Start Training -----')
    trainer.train()

    # Test
    test_model(test_dataframe, model)

    # Save
    trainer.save_model(OUTPUT_FILEPATH)
    feature_extractor.save_pretrained(OUTPUT_FILEPATH)
    logger.info(f'Saved to {OUTPUT_FILEPATH}')

if __name__ == '__main__':
    main()