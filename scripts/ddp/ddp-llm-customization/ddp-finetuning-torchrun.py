import os
import pip
import random
import numpy as np
from datasets import load_dataset

from transformers import get_scheduler
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

def setup_ddp(random_seed):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    init_process_group(backend="nccl")

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        test_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        local_rank: int,
    ) -> None:
        self.local_rank = local_rank
        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank)

    def run_epoch(self, max_epochs):
        num_training_steps = max_epochs * len(self.train_data)

        #学習
        for epoch in range(1, max_epochs + 1):
            print(f"[GPU{self.local_rank}] Epoch {epoch} | Steps: {len(self.train_data)}")
            self._train(num_training_steps)

            if epoch % 10 == 0:
                accuracy = self._test()
                print(f"Epoch: {epoch}, Accuracy: {accuracy}")
            scheduler.step()


    def _train(self, num_training_steps):
        self.model.train()

        #num_training_steps = num_epochs * len(train_loader) / world_size
        lr_scheduler = get_scheduler(name='linear', optimizer=self.optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        for batch in self.train_data:
            batch = {k: v.to(self.local_rank) for k, v in batch.items()}
            self.optimizer.zero_grad()
            outputs = self.model(**batch)

            loss = outputs.loss
            loss.backward()

            self.optimizer.step()
            lr_scheduler.step()

    def _test(self):
        self.model.eval()
    
        correct, total = 0, 0
        with torch.no_grad():
            for data in self.test_data:
                images, labels = data[0].to(self.local_rank), data[1].to(self.local_rank)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
        accuracy = correct / total
    
        return accuracy

def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    epochs = int(os.environ["EPOCHS"])
    batch_size = int(os.environ["BATCH_SIZE"])
    learning_rate = float(os.environ["LEARNING_RATE"])
    random_seed = int(os.environ["RANDOM_SEED"])
    save_every = 10

    setup_ddp(random_seed)

    # 1. モデル及びトークナイザーの準備
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
    ## Pytorchで扱えるように変換
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    
    # 2.データの準備
    dataset = load_dataset("yelp_review_full")
    dataset['train'][0]

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    ## text情報はモデルに入力しないため削除
    tokenized_datasets = tokenized_datasets.remove_columns(['text'])
    ## モデルでは引数がlabelsであると仮定されているので、labelカラムの名前をlabelsに変更
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    ## Pytorchに入力できるようにlistからtorchに変更
    tokenized_datasets.set_format('torch')

    train_dataset = tokenized_datasets["train"].shuffle()
    train_sampler = DistributedSampler(train_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True)
    train_data = DataLoader(train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        pin_memory=True,
        sampler=train_sampler)

    test_dataset = tokenized_datasets["test"].shuffle()
    test_data = DataLoader(dataset=test_dataset,
            batch_size=batch_size,
            num_workers=os.cpu_count(),
            pin_memory=True,
            shuffle=False)

    # 3. optizerの定義
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # 4. 学習
    trainer = Trainer(model, train_data, test_data, optimizer, local_rank)
    trainer.run_epoch(epochs)

    destroy_process_group()

if __name__=="__main__":
    main()