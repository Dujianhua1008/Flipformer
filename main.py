import os
import sys
import logging
import threading
import math
import time
import torch
import wandb
import torch.optim as optim
import argparse
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from utils import set_seed, compute_metrics
from datasets import Dataset,load_dataset
from transformers import BertConfig, LongformerConfig, BigBirdConfig, AutoTokenizer, Trainer, DataCollatorWithPadding, TrainingArguments
from models import FlipFormer, FastFormer, Transformer, BigBird, Longformer, AFT, Linformer, Performer
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
    

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=int, default=4)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--use_wandb', action="store_true")
    args=parser.parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    if args.use_wandb:
        run = wandb.init(
            project='flipformer',
            group=args.model_name,
            job_type=args.dataset_name
        )
        wandb.config = {
            "lr": args.lr,
            "batch_size" : args.train_batch_size,
            "model_name" : args.model_name,
            "dataset_name" : args.dataset_name,
        }
    # load dataset
    if args.dataset_name is not None:
        dataset = load_dataset(args.dataset_name)

    # prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    def process_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_length)
    tokenized_dataset = dataset.map(
        process_function, 
        batched=True, 
        remove_columns=['text'], 
        desc="Running tokenizer on dataset"
    )
    
    train_dataset = tokenized_dataset['train']
    eval_dataset  = tokenized_dataset['test']
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader  = DataLoader(train_dataset,shuffle=True,collate_fn=data_collator,batch_size=args.train_batch_size)
    eval_loader   = DataLoader(eval_dataset,collate_fn=data_collator,batch_size=args.eval_batch_size)
    # config model
    if args.model_name=='flipformer':
        config = BertConfig.from_json_file('config.json')
        config.num_labels = args.num_labels
        config.vocab_size = tokenizer.vocab_size
        model = FlipFormer(config)
    elif args.model_name=='fastformer':
        config = BertConfig.from_json_file('config.json')
        config.num_labels = args.num_labels
        config.vocab_size = tokenizer.vocab_size
        model = FastFormer(config)
    elif args.model_name=='longformer':
        config = LongformerConfig.from_json_file('config.json')
        config.num_labels = args.num_labels
        config.vocab_size = tokenizer.vocab_size
        config.max_position_embeddings += 3
        config.attention_window = [32,64]
        model = Longformer(config)
    elif args.model_name=='bigbird':
        config = BigBirdConfig.from_json_file('config.json')
        config.num_labels = args.num_labels
        config.vocab_size = tokenizer.vocab_size
        model = BigBird(config)
    elif args.model_name=='performer':
        config = BertConfig.from_json_file('config.json')
        config.num_labels = args.num_labels
        config.vocab_size = tokenizer.vocab_size
        model = Performer(config)
    elif args.model_name=='aft-simple':
        config = BertConfig.from_json_file('config.json')
        config.num_labels = args.num_labels
        config.vocab_size = tokenizer.vocab_size
        model = AFT(config)
    elif args.model_name=='linformer':
        config = BertConfig.from_json_file('config.json')
        config.num_labels = args.num_labels
        config.vocab_size = tokenizer.vocab_size
        model = Linformer(config, args.max_length, args.max_length//2)
    elif args.model_name=='transformer':
        config = BertConfig.from_json_file('config.json')
        config.num_labels = args.num_labels
        config.vocab_size = tokenizer.vocab_size
        model=Transformer(config)
    else:
        raise ValueError('no such model type')
    logger.info(model)
    # config optimizer
    optimizer = optim.Adam([{"params" : model.parameters(), "lr":args.lr}])
    device = torch.device('cuda:{}'.format(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device("cpu")
    # training the model
    model.to(device)
    set_seed(args.seed)
    cnt = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        for step, batch in enumerate(tqdm(train_loader)):
            batch.to(device)
            labels = batch['labels']
            loss, logits = model(batch['input_ids'], batch['labels'])
            total_loss += loss.detach().float()
            acc = compute_metrics(logits.detach().cpu(),labels.detach().cpu())
            total_acc += acc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt+=1
            if cnt % 30 == 0:
                if args.use_wandb:
                    run.log({"loss":total_loss/step})
            if cnt % 100 == 0:

                logger.info('loss:{} acc:{}'.format(total_loss/step,total_acc/step))
        model.eval()
        logits_list, labels_list = [], []
        for step, eval_batch in enumerate(tqdm(eval_loader)):
            eval_batch.to(device)
            labels = eval_batch['labels']
            loss, logits = model(eval_batch['input_ids'], eval_batch['labels'])
            logits_list.append(logits.detach().cpu())
            labels_list.append(labels.detach().cpu())
        logits_list, labels_list = np.argmax(torch.cat(logits_list).numpy(), axis=-1), torch.cat(labels_list).numpy()
        # acc = compute_metrics(logits, labels)
        acc = accuracy_score(labels_list,logits_list)
        logger.info('test_acc:{:.5f}'.format(acc))
        if args.use_wandb:
            run.summary['test/acc'] = acc
    if args.use_wandb:
        wandb.finish()
                   
if __name__=='__main__':
    main()