# Sentiment Model Testing
Here we will evaluate the accuracy of the sentiment model using `evaluate.py` file. Before running the file you need to provide:
1. `groundtruth_IEMOCAP.csv` - This file contains the true transcriptions as provided by IEMOCAP. 
    - However, the file provided is in `.log` format, and also nees to be correctly formatted before we can use it. 
    - You can use `notebooks/TranscriptionFormatter.ipynb` to covert `groundtruth_IEMOCAP.log` to `groundtruth_IEMOCAP.csv`.
    - Here is an example: `[2022-02-15 11:28:44,180][text2pickle.py][line:64][INFO] Ses01F_script01_3_F000: YOU'RE THE ONLY ONE WHO STILL LOVES HIS PARENTS` will be formatted to `Ses01F_script01_3_F000 YOU'RE THE ONLY ONE WHO STILL LOVES HIS PARENTS`
2. `sentiment_truths_IEMOCAP.csv` - This file contains the remapped emotions provided by IEMOCAP into Sentiment.
    - Example: `H` in the TSV file will be converted to `Positive`.
    - Therfore we map 4 emotions into 3 sentiments.
    - You can obtain this file using `notebooks/EmotionToSentimentFormatter.ipynb`.

# Evaluate.py Workflow
1. Setup (Import Libraries, Dataset, Filepath, etc)
2. Load true transcriptions from `groundtruth_IEMOCAP.csv`
3. Load sentiment labels from `sentiment_truths_IEMOCAP.csv`
4. Perform inference: 
    1. Apply NLP Preprocessing on true transcriptions
        - Expand Contractions
        - Text Lemmantization
        - Remove Numbers
        - Remove Stopwords
        - (Any or all functions can be selected. Check `./fold-x/.log` for which function is applied)
    2. Obtain sentiment using cleaned transcriptions as input
    3. Get TestUA score (Check `./fold-x/.log` for results)
5. Save outputs (transcription, sentiment, sentiment scores) as `predictions.csv`

# Usage
1. Change directory to which Sentiment model you want.
2. Run `evaluate.ipynb` for one fold. Ensure that code functions, and output is as expected.
3. Convert to `.py` file:
    - run `jupyter nbconvert --to script evaluate.ipynb`
    - open `evaluate.py` and change `PYTHON_SCRIPT` to `True`
4. Run `run.sh` script to obtain transcriptions of all 10 folds.
    - Each fold will run once and will generate a folder containing:
        1. `fold-x.logs` - Parameters, and TestUA scores
        2. `predictions.csv`
5. Run `compute_results.sh` script to obtain the average wer across all 10 folds.
    - `TestUA_results.txt` is generated containing results
    - `README.md` is generated with markdown table containing results for viewing on GitHub

# Notes on groundtruth_IEMOCAP
It is quite confusing, as there are 3 types of the same file:
1. `groundtruth_IEMOCAP.log` - This is the file downloaded from IEMOCAP directly.
    - Preview: `[2022-02-15 11:28:44,180][text2pickle.py][line:64][INFO] Ses01F_script01_3_F000: YOU'RE THE ONLY ONE WHO STILL LOVES HIS PARENTS`
2. `groundtruth_IEMOCAP.csv` - This is a formatted file for use with `evaluate.py` for **Sentiment Analysis** only. This is because for speech-to-text, we will get the transcripts by performing inference on the audio itself.
    - Preview: `Ses01F_script01_3_F000,YOU'RE THE ONLY ONE WHO STILL LOVES HIS PARENTS`
3. `groundtruth_IEMOCAP.txt` - This is a formatted version for use with `compute_wer.py` as that file wants `.txt` file only
    - Preview: `Ses01F_script01_3_F000 YOU'RE THE ONLY ONE WHO STILL LOVES HIS PARENTS`
