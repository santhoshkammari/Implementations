import trackio as wandb
import random
import time

runs = 1
epochs = 500


def simulate_multiple_runs():
    for run in range(runs):
        wandb.init(project="dummy-exp", config={
            "epochs": epochs,
            "learning_rate": 0.001,
            "batch_size": 64
        })

        for epoch in range(epochs):
            train_loss = random.uniform(0.2, 1.0)
            train_acc = random.uniform(0.6, 0.95)

            val_loss = train_loss - random.uniform(0.01, 0.1)
            val_acc = train_acc + random.uniform(0.01, 0.05)

            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })

            time.sleep(5)

    wandb.finish()


simulate_multiple_runs()