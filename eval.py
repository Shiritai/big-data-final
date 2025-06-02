import pandas as pd
from grader import score

submission = pd.read_csv("public_submission.csv").sort_values("id").reset_index(drop=True)
labels_pred = submission["label"].tolist()

print(f"Score: {score(labels_pred):.4f}")
