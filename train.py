import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataloader import SpO2Dataset, InferenceDataset, list_samples, subject_of
from engine import set_seed, train_epoch, validate, predict, write_csv
from loss import SpO2Loss
from model import CoDeitPhys


def run_fold(args, fold, paths, metadata):
    output = Path(args.out) / f'fold_{fold}'
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f'Choose an empty output folder: {output}')
    set_seed(args.seed)
    train_paths = [p for p in paths if subject_of(p, metadata) != fold]
    test_paths = [p for p in paths if subject_of(p, metadata) == fold]
    if len(train_paths) < 2 or not test_paths:
        raise ValueError('Need at least two training windows and non-empty held-out windows.')
    kwargs = dict(root_dir=args.data, name='pure', metadata=metadata, cache_in_ram=args.ram)
    train_data = SpO2Dataset(img_paths=train_paths, **kwargs)
    validation_data = SpO2Dataset(img_paths=test_paths, **kwargs)
    inference_data = InferenceDataset(img_paths=test_paths, **kwargs)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=len(train_data) % args.batch_size == 1)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size,
                                   shuffle=False, num_workers=args.workers)
    inference_loader = DataLoader(inference_data, batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.workers, generator=torch.Generator())
    device = torch.device(args.device)
    model = CoDeitPhys(args.deit_init, args.lambda3, args.lambda4).to(device)
    criterion = SpO2Loss(args.lambda1, args.lambda2)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_mae, best_epoch, history = float('inf'), 0, []


    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        scores = validate(model, validation_loader, device)
        history.append({'epoch': epoch, 'train_loss': train_loss, **scores})
        if scores['MAE'] < best_mae:
            best_mae, best_epoch = scores['MAE'], epoch
            torch.save(model.state_dict(), output / 'best_spo2_model.pt')
        scheduler.step()
        print(f'fold={fold} epoch={epoch}/{args.epochs} loss={train_loss:.6f} MAE={scores["MAE"]:.6f}', flush=True)
    torch.save(model.state_dict(), output / 'last_spo2_model.pt')
    write_csv(output / 'epochs.csv', history, list(history[0]))
    last_rows, last_scores = predict(model, inference_loader, device)
    write_csv(output / 'predictions_last.csv', last_rows)
    model.load_state_dict(torch.load(output / 'best_spo2_model.pt', map_location='cpu', weights_only=True), strict=True)
    best_rows, best_scores = predict(model, inference_loader, device)
    write_csv(output / 'predictions_best.csv', best_rows)
    report = {'seed': args.seed, 'fold': fold, 'epochs': args.epochs, 'best_epoch': best_epoch,
              'selection': 'minimum held-out subject MAE', 'best': best_scores, 'last': last_scores,
              'train_windows': len(train_paths), 'test_windows': len(test_paths),
              'lambda': [args.lambda1, args.lambda2, args.lambda3, args.lambda4],
              'arguments': vars(args)}
    (output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--manifest')
    parser.add_argument('--deit-init', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--fold', default='01', help='subject ID, or all for leave-one-subject-out')
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--ram', action='store_true')
    for i, default in enumerate([1.0, 1.0, 0.0, 1.0], 1):
        parser.add_argument(f'--lambda{i}', type=float, default=default)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.workers < 0 or args.threads < 1:
        parser.error('epochs/threads >=1, batch-size >=2, workers >=0 required')
    torch.set_num_threads(args.threads)
    paths, metadata = list_samples(args.data, args.manifest)
    subjects = sorted({subject_of(p, metadata) for p in paths})
    folds = subjects if args.fold == 'all' else [args.fold]
    if not set(folds).issubset(subjects):
        parser.error(f'Unknown fold. Available: {subjects}')
    reports = [run_fold(args, fold, paths, metadata) for fold in folds]
    (Path(args.out) / 'folds.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
