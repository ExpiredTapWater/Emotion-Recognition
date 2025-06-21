## Merge Folder

This folder contains all files related to running the merge program, including scripts, predictions, and model outputs.

### Folders

- **`Merge Results/`**  
  Contains output logs generated after running the merge program.

- **`Predict Results/`**  
  Contains predictions in CSV format, visualizations, and logs generated before merging.

- **`wav2vec2/`**  
  Stores IEMOCAP-trained model files for each fold, along with the training scripts.

### Scripts

- **`Predict.ipynb`**  
  Generates predictions from both the primary (speech) and secondary (text) pipelines, and formats the output for use in `merge.ipynb`. The outputs can be found in the `Predict Results` folder.

- **`merge.ipynb`**  
  Merges predictions using the outputs from `Predict.ipynb`. The outputs can be found in the `Merge Results` folder.
