import torch
import numpy as np


def train(epoch, model, loss_fn, train_loader, optimizer, device, run=None):
    model.train()

    for _, data in enumerate(train_loader, 0):
        ids = data["ids"].to(device)
        mask = data["mask"].to(device)
        token_type_ids = data["token_type_ids"].to(device)
        targets = data["targets"].to(device)
        outputs = model(ids, mask, token_type_ids)
        loss = loss_fn(outputs, targets)

        if _ % 500 == 0:
            print(f"Epoch: {epoch}, Loss:  {loss.item()}")

        acc = (
            np.sum(
                torch.sigmoid(outputs).cpu().detach().numpy().round()
                == targets.cpu().detach().numpy()
            )
        ) / targets.cpu().detach().numpy().size

        if run:
            run["train/batch/loss"].append(loss)
            run["train/batch/acc"].append(acc)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # grad descent step
        optimizer.step()
