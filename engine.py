import csv
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metrics(gt, pred):
    gt, pred = np.asarray(gt, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    error = pred - gt
    if not len(gt) or not np.isfinite(gt).all() or not np.isfinite(pred).all():
        raise ValueError('Empty or non-finite predictions.')
    return {
        'MAE': float(np.abs(error).mean()),
        'RMSE': float(np.sqrt(np.square(error).mean())),
        'MAPE': float((np.abs(error) / gt * 100).mean()) if np.all(gt > 0) else None,
        'PCC': float(np.corrcoef(gt, pred)[0, 1]) if len(gt) > 1 and gt.std() > 0 and pred.std() > 0 else None,
    }


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total, count = 0.0, 0
    for *views, labels in loader:
        views = [x.to(device) for x in views]
        labels = labels.to(device)
        optimizer.zero_grad()
        output, contrast = model(*views)
        loss = criterion(output.squeeze(1), labels, contrast)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss.')
        loss.backward()
        optimizer.step()
        total += loss.item() * len(labels)
        count += len(labels)
    if not count:
        raise ValueError('No training batches.')
    return total / count


@torch.inference_mode()
def validate(model, loader, device):
    model.eval()
    gt, pred = [], []
    for *views, labels in loader:
        output, _ = model(*[x.to(device) for x in views])
        gt.extend((labels * 20 + 80).tolist())
        pred.extend((output.squeeze(1) * 20 + 80).cpu().tolist())
    return metrics(gt, pred)


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    rows = []
    for image, mwli, gt, names in loader:
        output = model.predict(image.to(device), mwli.to(device)).squeeze(1)
        rows.extend(zip(names, gt.tolist(), (output * 20 + 80).cpu().tolist()))
    return rows, metrics([r[1] for r in rows], [r[2] for r in rows])


def write_csv(path, rows, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as file:
        if columns is None:
            writer = csv.writer(file)
            writer.writerow(['filename', 'gt', 'pred'])
        else:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
        writer.writerows(rows)
