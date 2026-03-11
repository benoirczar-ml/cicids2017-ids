import cupy as cp


def confusion_matrix_binary(y_true, y_pred):
    y_true = y_true.astype(cp.int32)
    y_pred = y_pred.astype(cp.int32)
    idx = y_true * 2 + y_pred
    counts = cp.bincount(idx, minlength=4)
    tn, fp, fn, tp = counts[0], counts[1], counts[2], counts[3]
    return int(tn), int(fp), int(fn), int(tp)


def precision_recall_f1(tn, fp, fn, tp):
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return float(precision), float(recall), float(f1)


def roc_auc(y_true, y_score):
    y_true = y_true.astype(cp.float32)
    y_score = y_score.astype(cp.float32)
    pos = y_true == 1
    n_pos = int(cp.sum(pos))
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    # rank-based AUC
    order = cp.argsort(y_score)
    ranks = cp.empty_like(order, dtype=cp.float32)
    ranks[order] = cp.arange(1, len(y_score) + 1, dtype=cp.float32)
    sum_ranks_pos = cp.sum(ranks[pos])
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg + 1e-9)
    return float(auc)


def average_precision(y_true, y_score):
    y_true = y_true.astype(cp.int32)
    y_score = y_score.astype(cp.float32)
    total_pos = int(cp.sum(y_true))
    if total_pos == 0:
        return None
    order = cp.argsort(-y_score)
    y_true_sorted = y_true[order]
    tp = cp.cumsum(y_true_sorted)
    fp = cp.cumsum(1 - y_true_sorted)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (total_pos + 1e-9)
    recall_prev = cp.concatenate([cp.array([0.0], dtype=cp.float32), recall[:-1]])
    ap = cp.sum((recall - recall_prev) * precision)
    return float(ap)
