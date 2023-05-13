#! pip install datasets
#! pip install transformers
#! pip install seqeval
#! pip install evaluate
#! pip install pandas
#! pip install mendelai-brat-parser

import pandas as pd
import os

path = os.path.join(os.getcwd(), "brat-project-final")
split_file = "train-test-split.csv"
df_split = pd.read_csv(split_file, sep=";", names=["ID", "SET"], header=0)
data_split = dict(df_split.values)

# Load and preprocess data
# Dataset I - Texto e TokenClassificados
# Dataset II - Texto, elemento, classificacao do elemento
# Dataset III - Texto, elemento 1 e 2, classicacao da relacao
# OBS: Anotacoes dos essays 98, 114, 182, 248 e 337 possuíam erros ortográficos que foram corrigidos a mao quando detectados
import re
from brat_parser import get_entities_relations_attributes_groups

def intersects(interval, matches_found):
  start = interval[0]
  end = interval[1]
  for i, match_found in enumerate(matches_found):
    found_start = match_found[0]
    found_end = match_found[1]
    # Intersecao
    if (end >= found_start and end < found_end) or (
        start >= found_start and start < found_end
    ):
      return True, i
  
  return False, None



def subfinder(mylist, key, num, matches_found, entities):
    element = entities[key].text
    pattern = re.findall(r"[\w']+|[.,!?;]", element)
    matches = list()
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and mylist[i:i+len(pattern)] == pattern:
          matches.append(list(range(i, i+len(pattern))))
    
    for match in matches:
        do_intersect, element = intersects(match, matches_found)
        if do_intersect and len(matches) > 1:
          continue
        elif not do_intersect:
          matches_found[num] = match
          break
        elif do_intersect and len(matches) == 1:
            matches_found[element] = [0, 0]
            matches_found[num] = match
            matches_found = subfinder(
                mylist, entities.key()[element], matches_found, entities
            )
            break
  
    return matches_found


def create_text_lists(text_id, entities, raw_text):
  # Load text and turn into a list
  text_list = re.findall(r"[\w']+|[.,!?;]", raw_text)

  # Create annotaded list
  matches = list()
  for i in range(len(entities.keys())):
    matches.append([0, 0])
  
  for num, key in enumerate(entities.keys()):
    matches = subfinder(text_list, key, num, matches, entities)

  an_text = [0]*len(text_list)
  for match in matches:
    for n in match:
      an_text[n] = 2
    an_text[match[0]] = 1
  return [text_id, text_list, an_text]


train_data_span = list()
test_data_span = list()
train_data_component = list()
test_data_component = list()
train_data_relation = list()
test_data_relation = list()

test_num_components = list()

for essay in data_split.keys():

  essay_an_path = os.path.join(path, essay + ".ann")
  essay_path = os.path.join(path, essay + ".txt")
  text_num = re.findall(r'\d+', essay)
  text_id = text_num[0]
  # Load Annotations
  entities, relations, attributes, groups = get_entities_relations_attributes_groups(essay_an_path)
  # Load text and turn into a list
  raw_text = open(essay_path, "r", encoding="utf-8").read()
  raw_text = raw_text.replace("\n", " ")
  text_lists = create_text_lists(text_id, entities, raw_text)
  # Save to train or test span sets
  if data_split[essay] == "TRAIN":
    train_data_span.append(text_lists)
  else:
    test_data_span.append(text_lists)
  
  # Save to train or test component sets
  for key in entities.keys():
    element = entities[key].text
    component = entities[key].type
    if component == "MajorClaim":
      component_num = 0
    elif component == "Claim":
      component_num = 1
    else:
      component_num = 2
    component_list = [text_id, element, raw_text, component_num]
    if data_split[essay] == "TRAIN":
      train_data_component.append(component_list)
    else:
      test_data_component.append(component_list)
  

  if data_split[essay] == "TEST":
     test_num_components.append(len(entities.keys()))
  
  # Save to train or test relation sets
  for key in relations.keys():
    row = relations[key]
    relation = row.type
    source_id = row.subj
    target_id = row.obj
    source_text = entities[source_id].text
    target_text = entities[target_id].text
    if relation == "supports":
      relation_num = 0
    else:
      relation_num = 1
    relation_list = [text_id, source_text, target_text, relation_num]
    if data_split[essay] == "TRAIN":
      train_data_relation.append(relation_list)
    else:
      test_data_relation.append(relation_list)

df_span_train = pd.DataFrame(train_data_span, columns = ["text_id", "tokens", "chunk_tags"])
df_span_test = pd.DataFrame(test_data_span, columns = ["text_id", "tokens", "chunk_tags"])
df_component_train = pd.DataFrame(train_data_component, columns = ["text_id", "component_tokens", "text_tokens", "labels"])
df_component_test = pd.DataFrame(test_data_component, columns = ["text_id", "component_tokens", "text_tokens", "labels"])
df_relation_train = pd.DataFrame(train_data_relation, columns = ["text_id", "source_tokens", "target_tokens", "labels"])
df_relation_test = pd.DataFrame(test_data_relation, columns =  ["text_id", "source_tokens", "target_tokens", "labels"])

import datasets
from datasets import Dataset, DatasetDict, Features
from datasets import load_dataset, load_metric, concatenate_datasets

# Create Dataset I for span detection

features = Features(
    (
      {
          "text_id": datasets.Value("int32"),
          "tokens": datasets.Sequence(datasets.Value("string")),
          "chunk_tags":datasets.Sequence(
              datasets.features.ClassLabel(
                  names=[
                      "O",
                      "B-ARG",
                      "I-ARG",
                  ]
              )
          )
      }
        )
  )

train_ds = Dataset.from_pandas(df_span_train, features=features)
test_ds = Dataset.from_pandas(df_span_test, features=features)
del df_span_train
del df_span_test
original_data_span = DatasetDict()
original_data_span["train"] = train_ds
original_data_span["test"] = test_ds

# Create Dataset II for component classificantion
features = Features(
    (
      {
          "text_id": datasets.Value("int32"),
          "component_tokens": datasets.Value("string"),
          "text_tokens": datasets.Value("string"),
          "labels": datasets.features.ClassLabel(
                  names=[
                      "MajorClaim",
                      "Claim",
                      "Premise",
                  ]
              )
      }
        )
  )
train_ds = Dataset.from_pandas(df_component_train, features=features)
test_ds = Dataset.from_pandas(df_component_test, features=features)
del df_component_train
del df_component_test
original_data_component = DatasetDict()
original_data_component["train"] = train_ds
original_data_component["test"] = test_ds

# Create Dataset III for relation classificantion
features = Features(
    (
      {
          "text_id": datasets.Value("int32"),
          "source_tokens": datasets.Value("string"),
          "target_tokens": datasets.Value("string"),
          "labels": datasets.features.ClassLabel(
                  names=[
                      "supports",
                      "attacks",
                  ]
              )
      }
        )
  )
train_ds = Dataset.from_pandas(df_relation_train, features=features)
test_ds = Dataset.from_pandas(df_relation_test, features=features)
del df_relation_train
del df_relation_test
original_data_relation = DatasetDict()
original_data_relation["train"] = train_ds
original_data_relation["test"] = test_ds

import evaluate
metric = evaluate.load("seqeval")
task_feature = original_data_span["train"].features["chunk_tags"]
label_names = task_feature.feature.names
data = original_data_span["train"].train_test_split(test_size=0.01)

id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}

#model_checkpoint = "xlm-roberta-base"
#model_checkpoint = "allenai/longformer-base-4096"
model_checkpoint = "bert-base-cased"
#model_checkpoint = 'bert-base-multilingual-cased'
#model_checkpoint= "bert-base-multilingual-uncased"
from transformers import (
    #RobertaTokenizerFast, RobertaForTokenClassification, 
    AutoModelForTokenClassification, AutoTokenizer,
    #BertTokenizerFast, BertForTokenClassification
    #LongformerForTokenClassification, LongformerTokenizerFast
    )

#tokenizer = RobertaTokenizerFast.from_pretrained(model_checkpoint, add_prefix_space=True)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
#model =  BertForTokenClassification.from_pretrained(
model =  AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)

import numpy as np


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], is_split_into_words=True, truncation=True,
    )
    all_labels = examples["chunk_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

from transformers import DataCollatorForTokenClassification
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
labels = data["train"][0]["chunk_tags"]

tokenized_training_input = data.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=data["train"].column_names,
)
tokenized_test_input = original_data_span.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=original_data_span["test"].column_names,
)

from transformers import TrainingArguments, Trainer
import torch
#torch.cuda.set_per_process_memory_fraction(2.0, 0)
torch.cuda.empty_cache()

filepath=os.getcwd()

args = TrainingArguments(
    output_dir=filepath,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    learning_rate=9.1e-5,
    weight_decay=0.01,
    seed=42,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit = 3
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_training_input["train"],
    eval_dataset=tokenized_training_input["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train pre-trained model
trainer.train()

# test_trainer = Trainer(model)
# Make prediction
trainer.evaluate(tokenized_test_input["test"])
#print(compute_metrics(raw_pred))
predictions = trainer.predict(tokenized_test_input["test"])
pred = np.argmax(predictions.predictions, axis=-1)
len(pred)

#list2 = [ elem for elem in tokenized_test_input["test"][0]["input_ids"] if elem not in (0, 2, -100)]
#len(list2)
true_components = list()
for_test = list()
total = 0
for i, labels in enumerate(list(tokenized_test_input["test"])):
  labels_list = [elem for elem in list(labels["labels"]) if elem != -100]
  former_label = 0
  start = -1
  end = -1
  for j, label in enumerate(labels_list):
    if label == 1 and former_label == 0:
      start = j
    elif label == 1 and former_label in (1,2):
      end = j-1
      component = [i, start, end]
      true_components.append(str(component))
      start = j
      end = -1
    elif label == 0 and former_label in (1,2):
      end = j-1
      component = [i, start, end]
      true_components.append(str(component))
      start = -1
      end = -1
    elif label == labels_list[-1] and start != -1:
      end = j
      component = [i, start, end]
      true_components.append(str(component))
      start = -1
      end = -1
    former_label = label
  if start != -1 and end == -1:
    print("ERRO")
  text_components = len(true_components) - total
  for_test.append(text_components)
  total = len(true_components)

predicted_components = list()
for i, labels in enumerate(list(pred)):
  labels_list = [elem for elem in list(labels) if elem != -100]
  labels_list.pop(0)
  former_label = 0
  start = -1
  end = -1
  for j, label in enumerate(labels_list):
    if label == 1 and former_label == 0:
      start = j
    elif label == 1 and former_label in (1,2):
      end = j-1
      component = [i, start, end]
      predicted_components.append(str(component))
      start = j
      end = -1
    elif label == 0 and former_label in (1,2):
      end = j-1
      component = [i, start, end]
      predicted_components.append(str(component))
      start = -1
      end = -1
    elif label == labels_list[-1] and start != -1:
      end = j
      component = [i, start, end]
      predicted_components.append(str(component))
      start = -1
      end = -1
    former_label = label
  if start != -1 or end != -1:
    print(f"ERRO: teste {i}")

print(len(test_num_components))
print(len(for_test))

for i in range(len(for_test)):
  if for_test[i] != test_num_components[i]:
    print(f"Texto: {test_data_span[i][0]}, Correto: {test_num_components[i]}, Visto: {for_test[i]}, {i}")

def intersection(lst1, lst2):
  return list(set(lst1) & set(lst2))

inter = len(intersection(true_components, predicted_components))

print(len(true_components))
print(len(predicted_components))
print(inter)

precision = inter/len(predicted_components)
recall = inter/len(true_components)
f1 = (2*precision*recall)/(precision+recall)

print("Precisão: ", precision)
print("Recall: ", recall)
print("F1: ", f1)