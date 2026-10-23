import argparse

import torch
from sklearn.metrics import classification_report, confusion_matrix

from dataset import get_loaders
from model import MLP

p = argparse.ArgumentParser()
p.add_argument("checkpoint")
p.add_argument("--shifted", action="store_true")
p.add_argument("--hidden", type=int, default=64)
args = p.parse_args()

_, _, test_loader = get_loaders(shifted=args.shifted)
model = MLP(args.hidden)
model.load_state_dict(torch.load(args.checkpoint))
model.eval()

preds, labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        preds.append(model(x).argmax(1))
        labels.append(y)
preds = torch.cat(preds).numpy()
labels = torch.cat(labels).numpy()

print(f"accuracy: {(preds == labels).mean():.4f}\n")
print(classification_report(labels, preds, digits=3))
print(confusion_matrix(labels, preds))
