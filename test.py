import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataloader import InferenceDataset, list_samples, subject_of
from engine import predict, write_csv
from model import CoDeitPhys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--manifest')
    parser.add_argument('--weights', required=True)
    parser.add_argument('--fold', default='01')
    parser.add_argument('--out', required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--ram', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    paths, metadata = list_samples(args.data, args.manifest)
    paths = [p for p in paths if args.fold == 'all' or subject_of(p, metadata) == args.fold]
    if not paths:
        parser.error('No windows selected.')
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'predictions.csv').exists() or (output / 'metrics.json').exists():
        raise FileExistsError('Choose an output folder without existing predictions/metrics.')
    dataset = InferenceDataset(args.data, paths, metadata=metadata, cache_in_ram=args.ram)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = CoDeitPhys().to(args.device)
    model.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True), strict=True)
    rows, scores = predict(model, loader, args.device)
    write_csv(output / 'predictions.csv', rows)
    (output / 'metrics.json').write_text(json.dumps({'windows': len(rows), **scores}, indent=2), encoding='utf-8')
    print(json.dumps({'windows': len(rows), **scores}), flush=True)


if __name__ == '__main__':
    main()
