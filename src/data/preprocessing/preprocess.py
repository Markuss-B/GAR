import numpy as np

def split_by_trainer(x, y, trainer_ids, train_ratio=0.8, val_ratio=0.2, seed=42):
    """Split data into train/val/test sets by trainer ID."""
    rng = np.random.RandomState(seed)
    
    trainers = np.unique(trainer_ids)
    shuffled = rng.permutation(trainers)
    
    n_train = int(len(trainers) * train_ratio)
    train_trainers = shuffled[:n_train]
    test_trainers = shuffled[n_train:]
    
    train_mask = np.isin(trainer_ids, train_trainers)
    test_mask = np.isin(trainer_ids, test_trainers)
    
    # Split train further into train/val
    train_idxs = np.where(train_mask)[0]
    rng.shuffle(train_idxs)
    
    n = len(train_idxs)
    split = int(n * (1 - val_ratio))
    
    train_idx = train_idxs[:split]
    val_idx = train_idxs[split:]
    
    return (
        x[train_idx], y[train_idx],
        x[val_idx], y[val_idx],
        x[test_mask], y[test_mask]
    )