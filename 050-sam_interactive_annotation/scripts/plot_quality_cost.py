import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/summary.csv")
plt.scatter(df["clicks"], df["iou"])
plt.xlabel("Clicks")
plt.ylabel("IoU")
plt.title("Quality vs Annotation Cost")
plt.show()
