import numpy as np

# Define the evaluation matrix
evaluation_matrix = np.array(
 [[13, 0, 0, 4 ,4, 0 ,0 ,1 ,2 ,0, 0],
  [0, 24, 1, 0 ,3, 0 ,1 ,0 ,2 ,1, 2],
  [12, 1, 5, 6 ,1, 0 ,1 ,2 ,2 ,0, 0],
  [3, 0, 2, 8 ,2, 0 ,1 ,2 ,0 ,3, 0],
  [0, 3, 0, 0 ,0, 0 ,1 ,0 ,0 ,0, 0],
  [0, 3, 0, 0 ,0, 0 ,1 ,0 ,0 ,0, 0],
  [0, 3, 0, 0 ,0, 0 ,1 ,0 ,0 ,0, 0],
  [0, 3, 0, 0 ,0, 0 ,1 ,0 ,0 ,0, 0],
  [0, 3, 0, 0 ,0, 0 ,1 ,0 ,0 ,0, 0],
  [0, 3, 0, 0 ,0, 0 ,1 ,0 ,0 ,0, 0],
  [0, 3, 0, 0 ,0, 0 ,1 ,0 ,0 ,0, 0]])

# Print the generated matrix
print(evaluation_matrix)
# Convert the values to percentages
evaluation_matrix = evaluation_matrix / 100

# Calculate the top-1 and top-5 accuracies
top_1_accuracy = np.diag(evaluation_matrix).mean() * 100
top_5_accuracy = np.mean(np.sum(evaluation_matrix[:, :5], axis=1)) * 100

# Print the results
print(f"Top-1 Accuracy: {top_1_accuracy}%")
print(f"Top-5 Accuracy: {top_5_accuracy}%")
