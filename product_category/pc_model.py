from argparse import ArgumentParser
from collections import Counter
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel
from transformers.optimization import AdamW


def mk_tensors(txt, tokenizer, max_seq_length):
    tok_res = tokenizer(
        txt, truncation=True, padding="max_length", max_length=max_seq_length
    )
    input_ids = tok_res["input_ids"]
    attention_mask = tok_res["attention_mask"]
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    return input_ids, attention_mask


def mk_ds(txt, tokenizer, max_seq_length, ys):
    input_ids, attention_mask = mk_tensors(txt, tokenizer, max_seq_length)
    return TensorDataset(input_ids, attention_mask, torch.tensor(ys))


class PCDataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name_or_path,
        max_seq_length,
        min_products_for_category,
        train_batch_size,
        val_batch_size,
        dataloader_num_workers,
        pin_memory,
        data_file_path=None,
        dataframe=None,
    ):
        super().__init__()
        self.data_file_path = data_file_path
        self.dataframe = dataframe
        self.min_products_for_category = min_products_for_category
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.pin_memory = pin_memory
        self.num_classes = None

    def setup(self, stage=None):
        if self.dataframe is None:
            self.dataframe = pd.read_csv(self.data_file_path)
            self.dataframe.dropna(inplace=True)

        cats = self.dataframe.category.apply(lambda x: x.split("|"))
        cat2cnt = Counter((j for i in cats for j in i))
        i2cat = sorted(
            k for k, v in cat2cnt.items() if v > self.min_products_for_category
        )
        cat2i = {v: k for k, v in enumerate(i2cat)}
        self.num_classes = len(i2cat)
        self.i2cat, self.cat2i = i2cat, cat2i

        ys = np.zeros((len(self.dataframe), len(i2cat)))
        for i, cats in enumerate(self.dataframe.category):
            idx_pos = [cat2i[cat] for cat in cats.split("|") if cat in cat2i]
            ys[i, idx_pos] = 1

        msk_val = self.dataframe.is_validation == 1
        self.df_trn = self.dataframe[~msk_val]
        self.df_val = self.dataframe[msk_val]
        idx_trn = np.where(~msk_val)[0]
        idx_val = np.where(msk_val)[0]
        self.ys_trn, self.ys_val = ys[idx_trn], ys[idx_val]

        self.train_dataset = mk_ds(
            list(self.df_trn.title), self.tokenizer, self.max_seq_length, self.ys_trn
        )
        self.eval_dataset = mk_ds(
            list(self.df_val.title), self.tokenizer, self.max_seq_length, self.ys_val
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.dataloader_num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.dataloader_num_workers,
            pin_memory=self.pin_memory,
        )


def getaccu(logits, ys):
    return ((logits > 0.0).int() == ys).float().mean()


class PCModel(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path,
        freeze_bert,
        num_classes,
        learning_rate,
        adam_beta1,
        adam_beta2,
        adam_epsilon,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_name_or_path = model_name_or_path
        self.bert = AutoModel.from_pretrained(self.model_name_or_path)
        self.freeze_bert = freeze_bert
        if self.freeze_bert == True:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.num_classes = num_classes
        self.W = nn.Linear(self.bert.config.hidden_size, self.num_classes)

    def forward(self, input_ids, attention_mask):
        h = self.bert(input_ids, attention_mask)["last_hidden_state"]
        h_cls = h[:, 0]
        return self.W(h_cls)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, ys = batch
        logits = self(input_ids, attention_mask)
        loss = F.binary_cross_entropy_with_logits(logits, ys)
        accu = getaccu(logits, ys)
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_accu", accu, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, ys = batch
        logits = self(input_ids, attention_mask)
        loss = F.binary_cross_entropy_with_logits(logits, ys)
        accu = getaccu(logits, ys)
        self.log("valid_loss", loss, on_step=False, sync_dist=True)
        self.log("valid_accu", accu, on_step=False, sync_dist=True)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            self.hparams.learning_rate,
            betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
            eps=self.hparams.adam_epsilon,
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=5e-5)
        parser.add_argument("--adam_beta1", type=float, default=0.9)
        parser.add_argument("--adam_beta2", type=float, default=0.999)
        parser.add_argument("--adam_epsilon", type=float, default=1e-8)
        return parser
