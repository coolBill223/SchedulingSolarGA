def fitness(y_pred, y_true):
    """
    Compare predicted and actual install duration in minutes.
    y_pred and y_true should both be real values (not scaled or torch).
    """
    ideal_upper = y_true + 120

    if y_pred < y_true:
        penalty = (y_true - y_pred) * 2
    elif y_pred > ideal_upper:
        penalty = (y_pred - ideal_upper) * 0.5
    else:
        penalty = 0

    return -penalty
