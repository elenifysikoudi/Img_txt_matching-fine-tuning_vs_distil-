This Github Repository contains experiments on Knowledge Distillation(KD) in comparison to a fine-tuned model on the task of image-text matching. It aims on exploring KD and its effectiveness.

# Dataset
The dataset used is a manipulation of MsCOCO captions, that gives a set of matching and mismatching image-caption pairs. The dataset is balanced and can be used for binary image-text matching classification.

In order to get access to the dataset you just need to run the file inside ```.\code\``` with this command:

```python3 create_dataset.py ```

# Experiments 
The folder ```.\code\``` contains four Jupyter notebooks using the aforementioned dataset.

```Data exploration ``` is just a notebook with some plots that gives information on how the data looks.

```Fine-tuned CLIP.ipynb```

This is the notebook containing the code to fine-tune CLIP and evaluate it on the dataset. CLIP is used as the teacher model on the knowledge distillation experiments.


```Baseline.ipynb```

This notebook contains the code to train and evaluate the student model without knowledge distillation and it serves as the baseline of the experiments.

```Distilation.ipynb```

Finally, this notebook contains the code to train  and evaluate the student model in distillation with the help of the teacher (fine-tuned CLIP). The set hyperparameters are the best configuration I could find.


