import torch
import numpy as np


def eval_model(validation_loader, model, loss_fn, device, run):
    losses = []
    correct_predictions = 0
    num_samples = 0
    # set model to eval mode (turn off dropout, fix batch norm)
    model.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(validation_loader, 0):

            if (batch_idx + 1) % 500 == 0:
                print(f"Batch: {batch_idx + 1}")

            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
            targets = data["targets"].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)

            loss = loss_fn(outputs, targets)
            losses.append(loss.item())

            # validation accuracy
            # add sigmoid, for the training sigmoid is in BCEWithLogitsLoss
            outputs = torch.sigmoid(outputs).cpu().detach().numpy().round()
            targets = targets.cpu().detach().numpy()
            correct_predictions += np.sum(outputs == targets)
            num_samples += targets.size  # total number of elements in the 2D array

    acc = float(correct_predictions) / num_samples
    loss = np.mean(losses)

    if run:
        run["val/loss"].append(loss)
        run["val/acc"].append(acc)

    return acc, loss
