# Speech-To-Text Model Testing
Here we will evaluate the accuracy of the speech to text model using `evaluate.py` file. Before running the file you need to check these files are there:
1. `compute_wer.sh` - This is the script to obtain WER average across 10 folds.
    - Please ensure this directory contains these files:
        1. `compute_wer.py` - This is the provided python file to compute WER based on two provided text files
        2. `groundtruth_IEMOCAP.txt` - This file contains correctly formatted transcriptions provided by IEMOCAP.
            - You can use `TranscriptionFormatter.ipynb` to covert `groundtruth_IEMOCAP.log` to `groundtruth_IEMOCAP.txt`.
    - Please ensure you have set these variables in the script to the correct filepath:
        1. COMPUTE_WER_PATH=".....\SIT-emotion-recognition\asr\compute_wer.py"
        2. GROUND_TRUTH_FILE=".....\SIT-emotion-recognition\asr\groundtruth_IEMOCAP.txt"
    - The script will use the current folder as base directory, so that you don't have to manually change the folder name when we test different models.

# Evaluate.py Workflow
1. Setup (Import Libraries, Dataset, Filepath, etc)
2. Perform inference: 
    1. Obtain transcriptions from the audio files
    2. Format/Clean transcriptions
        - Remove full stop
        - Remove all punctuations except apostrophe
        - Change to uppercase
        - (Logger will log whenever something is changed. Check `.log` for more details)
3. Save the transcriptions to `transcriptions.csv`.
4. Format `.csv` inton `.txt` for use later with WER calculation. Save as `transcriptions.txt`

# Usage
1. Change directory to which Speech-To-Text model you want.
2. Run `evaluate.ipynb` for one fold. Ensure that code functions, and output is as expected.
3. Convert to `.py` file:
    - run `jupyter nbconvert --to script evaluate.ipynb`
    - open `evaluate.py` and change `PYTHON_SCRIPT` to `True`
4. Run `run.sh` script to obtain transcriptions of all 10 folds. (`transcriptions.txt` is generated for each fold)
5. Run `compute_wer.sh` script to obtain the average wer across all 10 folds. (`wer_results.txt` is generated)
    - Ensure you have updated the filepath to `compute_wer.py` and `groundtruth_IEMOCAP.txt` inside the script.
    - You can use `notebook/FormatTranscriptions.ipynb` to convert the downloaded `.log` file to `.txt`.

# Notes on groundtruth_IEMOCAP
It is quite confusing, as there are 3 types of the same file:
1. `groundtruth_IEMOCAP.log` - This is the file downloaded from IEMOCAP directly.
    - Preview: `[2022-02-15 11:28:44,180][text2pickle.py][line:64][INFO] Ses01F_script01_3_F000: YOU'RE THE ONLY ONE WHO STILL LOVES HIS PARENTS`
2. `groundtruth_IEMOCAP.csv` - This is a formatted file for use with `evaluate.py` for **Sentiment Analysis** only. This is because for speech-to-text, we will get the transcripts by performing inference on the audio itself.
    - Preview: `Ses01F_script01_3_F000,YOU'RE THE ONLY ONE WHO STILL LOVES HIS PARENTS`
3. `groundtruth_IEMOCAP.txt` - This is a formatted version for use with `compute_wer.py` as that file wants `.txt` file only
    - Preview: `Ses01F_script01_3_F000 YOU'RE THE ONLY ONE WHO STILL LOVES HIS PARENTS`
