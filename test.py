from data.dataloader import load_data
from fitness import fitness

y_true = 540

#here we are testing the fitness score for different errors
#later put the real data into the model and test the fitness score
for error in [-50, 0, 30, 15, -124]:
    y_pred = y_true + error
    score = fitness(y_pred, y_true)
    print(f"Δ = {error:.2f} → Fitness Score = {score:.2f}")