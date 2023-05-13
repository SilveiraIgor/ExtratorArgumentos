! pip install datasets
! pip install transformers
! pip install seqeval
! pip install evaluate
! pip install pandas
! pip install mendelai-brat-parser

import pandas as pd
import os

#split_file = "gdrive/MyDrive/Doutorado/Argument Mining/ArgumentAnnotatedEssays-2.0/train-test-split.csv"
split_file = "train-test-split.csv"
df_split = pd.read_csv(split_file, sep=";", names=["ID", "SET"], header=0)
data_split = dict(df_split.values)
print(os.path)