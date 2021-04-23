import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataloader
import wandb
from model import ScenarioModel
from random_seed import RANDOM_SEED


np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

USE_WANDB = True

if USE_WANDB:
    wandb.init(project="argo_autoencoder_real")
    params = wandb.config
else:
    params = {}


params["round_name"] = "round2"
params["batch_size"] = 4  # 256 * 3
params["lr"] = 0.0001
params["momentum"] = 0.9
params["epochs"] = 800
params["sched_step_every_n_epochs"] = 2
params["sched_mult_lr_by_gamma_everystep"] = 0.85
# tb_writer = SummaryWriter(log_dir=f"runs/{params['round_name']}")
logging.basicConfig(
    level=logging.INFO
)  # ,filename=f"{round_name}log.txt",filemode="w")
if USE_WANDB:
    wandb.save("models.py")


def main():
    BATCH_SIZE = params["batch_size"]
    LEARNING_RATE = params["lr"]
    MOMENTUM = params["momentum"]
    num_epochs = params["epochs"]
    LOAD_MODEL = False
    sched_step_every_n_epochs = params["sched_step_every_n_epochs"]
    sched_mult_lr_by_gamma_everystep = params["sched_mult_lr_by_gamma_everystep"]

    # MODEL_FILE= "./epoch_0"
    logging.info("Loading datasets...")

    # only_real=True ensures that we only take the nominal data and not the anomalous one
    train_loader, dev_loader = dataloader.get_train_test_loaders(
        batch_size=BATCH_SIZE, only_real=True
    )

    model = nn.DataParallel(ScenarioModel())

    if torch.cuda.is_available():
        model = model.cuda()
    logging.info(model)
    if LOAD_MODEL:
        model.load_state_dict(torch.load(MODEL_FILE))

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=5e-5
    )

    sched = torch.optim.lr_scheduler.StepLR(
        optimizer, sched_step_every_n_epochs, gamma=sched_mult_lr_by_gamma_everystep
    )
    if USE_WANDB:
        wandb.watch(model, log="all")
    # sched = torch.optim.lr_scheduler.CyclicLR(optimizer,0.0001,0.001,5, cycle_momentum=False)
    logging.info("Starting Training")
    running_loss = 0
    for epoch in range(num_epochs):
        logging.info(f"Epoch:{epoch} LR:{sched.get_last_lr()}")
        # wandb.log({"CurrLR": sched.get_last_lr(), "Epoch": epoch})
        train_acc = train(
            train_loader, model, criterion, optimizer, epoch, running_loss
        )
        val_acc = evaluate(dev_loader, model, criterion, optimizer, epoch)
        # sched.step()
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"epoch_{epoch}.pth")
            if USE_WANDB:
                wandb.save(f"epoch_{epoch}.pth")
    torch.save(model.state_dict(), f"epoch_{epoch}.pth")
    if USE_WANDB:
        wandb.save(f"epoch_{epoch}.pth")


def train(train_loader, model, criterion, optimizer, epoch, running_loss):
    model.train()
    print_frequency = 10000
    all_outputs = []
    all_targets = []
    base_steps = epoch * len(train_loader.dataset)
    loss_agg = 0
    losses = []
    for i, (input) in tqdm(enumerate(train_loader), total=len(train_loader)):
        if torch.cuda.is_available():
            input = input.type(torch.FloatTensor).cuda()
            # target = target.type(torch.FloatTensor).cuda()

        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, input)
        loss.backward()
        optimizer.step()
        # batch_accuracy = calc_accuracy(output, target)
        batch_size = input.shape[0]
        losses.append(loss.item())
        loss_agg += loss.item() * batch_size
        running_loss = loss_agg / ((i + 1) * batch_size)
        if USE_WANDB:
            wandb.log(
                {"loss": loss.item(), "running_loss": running_loss, "epoch": epoch,}
            )

        if i % print_frequency == 0:
            logging.info(
                f"Epoch:{epoch} | Step:{i}/{len(train_loader)} | Loss: {loss.item()}"  # " | Batch Acc:{batch_accuracy}"
            )
        del input
        del loss

    # accuracy = accuracy_score(all_targets, all_outputs)
    logging.info(f"Epoch: {epoch} | Running Loss:{running_loss}")
    if USE_WANDB:
        wandb.log({"Running Loss": running_loss, "epoch": epoch})
    return running_loss


def evaluate(dev_loader, model, criterion, optimizer, epoch):
    with torch.no_grad():
        model.eval()
        losses = []
        for i, input in tqdm(enumerate(dev_loader), total=len(dev_loader)):
            if torch.cuda.is_available():
                input = input.type(torch.FloatTensor).cuda()

            output = model(input)
            loss = criterion(output, input)
            losses.append(loss.item())

            del input
        net_loss = np.array(losses).mean()
        logging.info(f"Epoch: {epoch} | Val Loss: {net_loss}")
        if USE_WANDB:
            wandb.log({"Val Loss": net_loss, "epoch": epoch})
    return net_loss


if __name__ == "__main__":

    main()
