import os
import sys
import random
import numpy as np

import torch
import torchvision
from torchvision import transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torch.distributed import init_process_group, destroy_process_group

def setup_ddp(args, gpu_id, world_size):
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    init_process_group(backend='nccl', rank=gpu_id, world_size=world_size)

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
        test_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        # save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        # self.save_every = save_every
        self.model = DDP(model, device_ids=[self.gpu_id], output_device=self.gpu_id)

    def run_epoch(self, max_epochs):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        b_sz = len(next(iter(self.train_data))[0])
        for epoch in range(1, max_epochs + 1):
            print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
            self._train()

            if epoch % 10 == 0:
                accuracy = self._test()
                print(f"Epoch: {epoch}, Accuracy: {accuracy}")
            scheduler.step()

    def _train(self):
        self.model.train()
    
        criterion = nn.CrossEntropyLoss()
    
        for data in self.train_data:
            inputs, labels = data[0].to(self.gpu_id), data[1].to(self.gpu_id)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
    
    def _test(self):
        self.model.eval()
    
        correct, total = 0, 0
        with torch.no_grad():
            for data in self.test_data:
                images, labels = data[0].to(self.gpu_id), data[1].to(self.gpu_id)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
        accuracy = correct / total
    
        return accuracy

def main(args):
    gpu_id = int(os.environ["LOCAL_RANK"])
    world_size = torch.cuda.device_count()
    
    setup_ddp(args , gpu_id, world_size)

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
 
    training_data_set = torchvision.datasets.CIFAR10(root="/tmp/data",train=True, download=True, transform=transform)
    training_sampler = DistributedSampler(dataset=training_data_set)
    train_data = DataLoader(dataset=training_data_set,
        batch_size=args.batch_size,
        num_workers=os.cpu_count(),
        pin_memory=True,
        sampler=training_sampler)

    test_data_set = torchvision.datasets.CIFAR10(root="/tmp/data", train=False, download=True, transform=transform)
    test_data = DataLoader(dataset=test_data_set,
        batch_size=args.batch_size,
        num_workers=os.cpu_count(),
        pin_memory=True,
        shuffle=False)
 
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(test_data_set.classes))
    # model = Net() 

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)

    trainer = Trainer(model, train_data, test_data, optimizer, gpu_id)
    trainer.run_epoch(args.epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--pre-trained", action="store_true")

    args = parser.parse_args()

    main(args)