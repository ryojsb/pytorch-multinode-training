import os
import sys
import tempfile
import torch
import torchvision
from torchvision import transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5554'

    # initialize the process group
    # init_process_group(backend='gloo', rank=rank, world_size=world_size)
    init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])

    # Epochの実行
    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)

        for source, targets in self.train_data:
            # 学習データ(source)とラベル(target)をそれぞれGPU上に乗せる
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            # 学習の実行
            self._run_batch(source, targets)

    # Modelの学習及びBackpropagation(誤差逆伝播)の実施
    def _run_batch(self, source, targets):
        # 勾配の値を正しく計算するため、パラメータの勾配を0に設定する
        self.optimizer.zero_grad()
        # Modelの学習
        output = self.model(source)
        # Backpropagationの実行
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "/tmp/checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)

    data = torchvision.datasets.CIFAR10(root="/tmp/data", train=True, download=True, transform=transforms.ToTensor())
    sampler = DistributedSampler(data, rank=rank)
    train_data = DataLoader(data, batch_size=64, pin_memory=True, shuffle=False, sampler=sampler)

    model = Net()    
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    # trainer = Trainer(model, train_data, optimizer, "cuda", save_every)
    trainer.train(total_epochs)
    destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)