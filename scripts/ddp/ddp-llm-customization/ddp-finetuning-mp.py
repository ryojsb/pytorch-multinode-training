import pip
pip.main(['install', 'datasets'])

from datasets import load_dataset
from transformers import get_scheduler
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

from tqdm.auto import tqdm

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import os

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


def train(rank, world_size, small_train_dataset):
    # refer to https://pytorch.org/docs/master/notes/ddp.html
    # DDPの利用にはdist.init_process_groupで初期化する必要あり
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # モデルの作成
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
    model = model.to(rank)
    # DDP用のモデルの作成
    model = DDP(model, device_ids=[rank])

    #DDP用のサンプラーの作成
    ## これを使うっことによりサンプルをプロセスごとにうまく配分してくれるらしい
    train_sampler = DistributedSampler(small_train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True)
    train_loader = DataLoader(small_train_dataset,
        batch_size=32,
        shuffle=train_sampler is None,
        pin_memory=True,
        sampler=train_sampler)

    ## optizerとschedulerの定義
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    #num_training_steps = num_epochs * len(train_loader) / world_size
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(name='linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    #学習
    progress_bar = tqdm(range(int(num_training_steps + 1)))
    model.train()
    for epoch in range(num_epochs):
        # データの順序を帰るためにepochごとにset_epochをする必要あり
        train_sampler.set_epoch(epoch)
        for batch in train_loader:
            batch = {k: v.to(rank) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

def main():
    # 1.データの準備
    ## データのダウンロード
    dataset = load_dataset("yelp_review_full")
    dataset['train'][0]

    ## Pytorchで扱えるように変換
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # text情報はモデルに入力しないため削除
    tokenized_datasets = tokenized_datasets.remove_columns(['text'])
    # モデルでは引数がlabelsであると仮定されているので、labelカラムの名前をlabelsに変更
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    # Pytorchに入力できるようにlistからtorchに変更
    tokenized_datasets.set_format('torch')

    # データ量が多いため一部のみ利用
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42)

    # 2. DDPを利用して学習
    # DDPを利用するには環境変数MASTER_ADDRとMASTER_ADDRを設定する必要がある
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    # 3. TDB
    

if __name__=="__main__":
    n_gpus = 4
    world_size = n_gpus
    mp.spawn(main,
        args=(world_size,small_train_dataset,),
        nprocs=world_size,
        join=True)