import os
from subprocess import PIPE
import subprocess
import time
from typing import Dict
import torch
import numpy as np
from sklearn import metrics
from collections import defaultdict
import logging

def labels_to_multihot(labels, num_classes=146):
    multihot_labels = torch.zeros(len(labels), num_classes)
    for i, label in enumerate(labels):
        for l in label:
            multihot_labels[i][l] = 1
    return multihot_labels


def get_precision_recall_f1(y_true: np.array, y_pred: np.array, average='micro'):
    precision = metrics.precision_score(
        y_true, y_pred, average=average, zero_division=0)
    recall = metrics.recall_score(
        y_true, y_pred, average=average, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, average=average, zero_division=0)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    return accuracy, precision, recall, f1

def get_precision_recall_f1_curve(y_true: np.array, y_pred: np.array):
    result = defaultdict(list)
    for threshold in np.linspace(0, 0.5, 51):
        result['threshold'].append(threshold)
        tmp_y_pred = np.array(y_pred >= threshold, np.int64)

        p, r, f1_micro = get_precision_recall_f1(y_true, tmp_y_pred, 'micro')
        result['precision_micro'].append(round(p, 4))
        result['recall_micro'].append(round(r, 4))
        result['f1_micro'].append(round(f1_micro, 4))

        p, r, f1_macro = get_precision_recall_f1(y_true, tmp_y_pred, 'macro')
        result['precision_macro'].append(round(p, 4))
        result['recall_macro'].append(round(r, 4))
        result['f1_macro'].append(round(f1_macro, 4))

        result['f1_avg'].append(round((f1_micro+f1_macro)/2, 4))
    return result


def evaluate(valid_dataloader, tokenizer, model, device, args):
    model.eval()
    all_predictions_accu = []
    all_predictions_law = []
    all_predictions_term = []
    all_labels_accu = []
    all_labels_law = []
    all_labels_term = []
    for i, data in enumerate(valid_dataloader):
        facts, labels_accu, labels_law, labels_term = data

        # move data to device
        labels_accu = torch.from_numpy(np.array(labels_accu)).to(device)
        labels_law = torch.from_numpy(np.array(labels_law)).to(device)
        labels_term = torch.from_numpy(np.array(labels_term)).to(device)
     
        with torch.no_grad():
            # forward
            logits_accu, logits_law, logits_term = model(facts, tokenizer)

        all_predictions_accu.append(logits_accu.softmax(dim=1).detach().cpu())
        all_labels_accu.append(labels_accu.cpu())
        all_predictions_law.append(logits_law.softmax(dim=1).detach().cpu())
        all_labels_law.append(labels_law.cpu())
        all_predictions_term.append(logits_term.softmax(dim=1).detach().cpu())
        all_labels_term.append(labels_term.cpu())

    all_predictions_accu = torch.cat(all_predictions_accu, dim=0).numpy()
    all_labels_accu = torch.cat(all_labels_accu, dim=0).numpy()
    all_predictions_law = torch.cat(all_predictions_law, dim=0).numpy()
    all_labels_law = torch.cat(all_labels_law, dim=0).numpy()
    all_predictions_term = torch.cat(all_predictions_term, dim=0).numpy()
    all_labels_term = torch.cat(all_labels_term, dim=0).numpy()
    
    accuracy_accu, p_macro_accu, r_macro_accu, f1_macro_accu = get_precision_recall_f1(all_labels_accu, np.argmax(all_predictions_accu, axis=1), 'macro')
    accuracy_law, p_macro_law, r_macro_law, f1_macro_law = get_precision_recall_f1(all_labels_law, np.argmax(all_predictions_law, axis=1), 'macro')
    accuracy_term, p_macro_term, r_macro_term, f1_macro_term = get_precision_recall_f1(all_labels_term, np.argmax(all_predictions_term, axis=1), 'macro')
    return accuracy_accu, p_macro_accu, r_macro_accu, f1_macro_accu, accuracy_law, p_macro_law, r_macro_law, f1_macro_law, accuracy_term, p_macro_term, r_macro_term, f1_macro_term

def get_gpu_memory_map() -> Dict[str, int]:
    """Get the current gpu usage.

    Return:
        A dictionary in which the keys are device ids as integers and
        values are memory usage as integers in MB.
    """
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader",],
        encoding="utf-8",
        # capture_output=True,          # valid for python version >=3.7
        stdout=PIPE,
        stderr=PIPE,  # for backward compatibility with python version 3.6
        check=True,
    )
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.stdout.strip().split(os.linesep)]
    gpu_memory_map = {f"{index}": memory for index, memory in enumerate(gpu_memory)}
    return gpu_memory_map 

def get_cuda_ids_with_sufficient_memory(memory_required=20000, max_memory=24*1024, cadidates_gpu_ids=['0', '1', '2', '3'], wait_until_done=True):
    """Get a CUDA ID with sufficient memory.

    Return:
        CUDA ID with form like f"cuda:{id}"
    """
    if wait_until_done:
        while True:
            cuda_ids = [cuda_id for cuda_id, memory in get_gpu_memory_map().items() if cuda_id in cadidates_gpu_ids and (memory < (max_memory - memory_required))]
            if cuda_ids:
                return cuda_ids
            logging.info('Waiting for GPUs with sufficient memory...')
            time.sleep(3600)
    else:
        cuda_ids = [cuda_id for cuda_id, memory in get_gpu_memory_map().items() if cuda_id in cadidates_gpu_ids and (memory < (max_memory - memory_required))]
        if cuda_ids:
            return cuda_ids
    return None

