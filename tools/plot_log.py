#!/usr/bin/python

import matplotlib.pyplot as plt
import pandas as pd
import sys

# Parse arguments
if len(sys.argv) < 2:
    print("Usage: " + sys.argv[0] + " loss_file")
    exit()
else:
    loss_file = sys.argv[1]

train_log = pd.read_csv(loss_file)
plt.plot(train_log["NumIters"], train_log["TrainingLoss"], 'g')
plt.xlabel('Iteration')
plt.xlabel('Train Loss')
plt.savefig('loss.png')

