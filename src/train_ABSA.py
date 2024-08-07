'''
Author: Yifei Wang
Github: ephiewangyf@gmail.com
Date: 2024-05-25 12:12:07
LastEditors: ephie && ephiewangyf@gmail.com
LastEditTime: 2024-05-27 22:00:39
FilePath: /Aspect-Based-Sentiment-Analysis/src/train_ABSA.py
Description: 
'''
from consts import *
import fire
import torch
import pandas as pd
import warnings
import sys
sys.path.insert(1, '../IMDB')
warnings.filterwarnings("ignore")


def train(work_type, batch=16, epochs=3, lr=3*1e-5, lr_schedule=False, adapter=True):
    """Train the model.

     Args:
         work_type (str): Training for which task choices: ['ABTE', 'ABSA']
         batch (int): Batch size for training (default: 5).
         epochs (int): Number of training epochs (default: 8).
         lr (float): Learning rate (default: 3*1e-5).
         lr_schedule (bool): Whether to use learning rate scheduling (default: False).
         adapter (bool): Whether to use Adapter(default: True).
     """
    if type(lr) == str:
        lr = eval(lr)
    # load
    data = pd.read_csv(TRAIN_DATA_PATH)

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")  # cuda GPU
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")  # check mac mps
    else:
        DEVICE = torch.device("cpu")  # otherwise cpu
    print(f"Using device: {DEVICE}")
    print(f"Learning rate: {lr}, Batch size: {batch}, Epochs: {epochs}")
    if work_type == 'ABTE':
        from abte import ABTEModel
        modelABTE = ABTEModel(tokenizer, adapter)
        modelABTE.train(data, batch_size=batch, lr=lr,
                        epochs=epochs, device=DEVICE, lr_schedule=lr_schedule)
    elif work_type == 'ABSA':
        from absa import ABSAModel
        modelABSA = ABSAModel(tokenizer, adapter=adapter)
        modelABSA.train(data, batch_size=batch, lr=lr,
                        epochs=epochs, device=DEVICE, lr_schedule=lr_schedule)
    else:
        raise Exception('wrong work_type')


if __name__ == '__main__':
    fire.Fire(train)
    print('Done')
