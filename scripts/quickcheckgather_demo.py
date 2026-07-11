import torch

# Simulate a batch of 4 samples, 5 classes
logits = torch.randn(4, 5)                # model output
pred   = logits.argmax(dim=1)             # predicted class indices, shape (4,)
# Example partial-label targets (each row sums to 1, non‑zero entries = allowed classes)
targets = torch.tensor([
    [0.0, 1.0, 0.0, 0.0, 0.0],   # only class 1 allowed
    [0.5, 0.5, 0.0, 0.0, 0.0],   # classes 0 or 1 allowed
    [0.0, 0.0, 0.0, 1.0, 0.0],   # only class 3 allowed
    [0.0, 0.0, 0.33, 0.33, 0.34] # classes 2,3,4 allowed
])

# Boolean mask: True where the target allows the class
support = targets > 0                # shape (4,5), dtype=bool

# Original line (using gather):
cor_orig = support.gather(1, pred.unsqueeze(1)).sum().item()

# Alternative 1: advanced indexing (often clearer)
cor_alt1 = support[torch.arange(len(pred)), pred].sum().item()

# Alternative 2: one‑hot mask + logical AND
cor_alt2 = (support & torch.nn.functional.one_hot(pred, num_classes=5)).sum().item()

print(f"pred: {pred.tolist()}")
print(f"support:\n{support.int()}")
print(f"cor (gather)   : {cor_orig}")
print(f"cor (indexing) : {cor_alt1}")
print(f"cor (one‑hot)  : {cor_alt2}")
