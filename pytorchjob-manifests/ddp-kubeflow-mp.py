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

def setup_ddp(random_seed):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    init_process_group(backend="nccl")

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
        local_rank: int,
        # save_every: int,
    ) -> None:
        self.local_rank = local_rank
        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        # self.save_every = save_every
        self.model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank)

    def run_epoch(self, max_epochs):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        b_sz = len(next(iter(self.train_data))[0])
        for epoch in range(1, max_epochs + 1):
            print(f"[GPU{self.local_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
            self._train()

            if epoch % 10 == 0:
                accuracy = self._test()
                print(f"Epoch: {epoch}, Accuracy: {accuracy}")
            scheduler.step()

    def _train(self):
        self.model.train()
    
        criterion = nn.CrossEntropyLoss()
    
        for data in self.train_data:
            inputs, labels = data[0].to(self.local_rank), data[1].to(self.local_rank)
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
                images, labels = data[0].to(self.local_rank), data[1].to(self.local_rank)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
        accuracy = correct / total
    
        return accuracy

def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    epochs = int(os.environ["EPOCHS"])
    batch_size = int(os.environ["BATCH_SIZE"])
    learning_rate = float(os.environ["LEARNING_RATE"])
    random_seed = int(os.environ["RANDOM_SEED"])
    save_every = 10

    setup_ddp(random_seed)

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
 
    training_data_set = torchvision.datasets.CIFAR10(root="/tmp/data",train=True, download=True, transform=transform)
    training_sampler = DistributedSampler(dataset=training_data_set)
    train_data = DataLoader(dataset=training_data_set,
        batch_size=batch_size,
        num_workers=os.cpu_count(),
        pin_memory=True,
        sampler=training_sampler)

    test_data_set = torchvision.datasets.CIFAR10(root="/tmp/data", train=False, download=True, transform=transform)
    test_data = DataLoader(dataset=test_data_set,
        batch_size=batch_size,
        num_workers=os.cpu_count(),
        pin_memory=True,
        shuffle=False)
 
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(test_data_set.classes))
    # model = Net() 

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)

    trainer = Trainer(model, train_data, test_data, optimizer, local_rank)
    trainer.run_epoch(epochs)
    destroy_process_group()

if __name__ == "__main__":
    main()