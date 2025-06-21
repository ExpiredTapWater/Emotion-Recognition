## wav2vec2

The results in the logs have been updated to the full run.

Summary:

| Fold | CNN                       | Wav2vec2                  | Add S2T & Sentiment*   | Prediction Strategy           | Sentiment Strategy           | Parameters                   |
|------|---------------------------|---------------------------|------------------------|-------------------------------|------------------------------|------------------------------|
|      | TestUA / Best Epoch       | TestUA / Best Epoch       | TestUA                 | Strategy                      | Strategy / Threshold         | Flip / Metric / Threshold    |
| 0    | 0.4093 / 19               | 0.7103 / 3                | 0.7209                 | 'ignore-when-match'           | 'simple' / 0.76              | False / Argmax /  0.474      |
| 1    | 0.3805 / 3                | 0.7085 / 3                | 0.7295                 | 'ignore-when-match'           | 'simple' / 0.52              | True  / Argmax /  0.512      |
| 2    | 0.4998 / 21               | 0.6788 / 3                | 0.6989                 | 'ignore-when-match'           | 'refer'  / 0.30              | True  / Argmax /  0.562      |
| 3    | 0.5049 / 21               | 0.7079 / 3                | 0.7186                 | 'ignore-when-match'           | 'refer'  / 0.30              | True  / Argmax /  0.464      |
| 4    | 0.4984 / 66               | 0.6257 / 2                | 0.6603                 | 'ignore-when-match'           | 'simple' / 0.50              | True / Entrophy / 0.98       |
| 5    | 0.4813 / 48               | 0.6241 / 3                | 0.6319                 | 'ignore-when-match'           | 'simple' / 0.30              | False / Argmax /  0.536      |
| 6    | 0.5948 / 60               | 0.6123 / 2                | 0.6305                 | 'ignore-when-match'           | 'simple' / 0.80              | True  / Argmax /  0.452      |
| 7    | 0.5275 / 50               | 0.6661 / 3                | 0.6672                 | 'ignore-when-match'           | 'refer'  / 0.30              | True / Entrophy / 0.90       |
| 8    | 0.5824 / 65               | 0.6443 / 2                | Did not improve        | -                             | -                            | -                            |
| 9    | 0.5832 / 54               | 0.5593 / 2                | 0.5827                 | 'ignore-when-match'           | 'simple' / 0.36              | False / Entrophy / 0.90      |
| **Average** | **0.5062 / -**     | **0.6537 / -**            | **0.6711 / -**         | -                             | -                            | -                            |

*S2T = `facebook/wav2vec2-base-960h`, Sentiment = `cardiffnlp/twitter-roberta-base-sentiment-latest`

### In Progress
- Add in remaining arguments (optimizer, loss_type, etc)
- Format and add comments to code
- Add in Speech-To-Text using Gemini/ChatGPT
- Check possible bug in code: I test `ignore-when-match first`. So if `ignore-when-match` and `refer` has the same improvement, the code will say `ignore-when-match` is better.

### Completed
- Initial first working code. Converted small-project notebook to regular python. Able to use run.sh to get 10 folds results
- Run full Epoch (E=100, EarlyStopping=20)
- Add in original S2T and Sentiment results


