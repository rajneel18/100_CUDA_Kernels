import numpy as np
import pandas as pd
# Load the dataset (u.data file)
file_path = "ml-100k/u.data"
data = pd.read_csv(file_path, sep="\t", names=["user_id", "item_id", "rating", "timestamp"])

# Number of users and items (MovieLens 100k has 943 users and 1682 items)
num_users = data["user_id"].max()
num_items = data["item_id"].max()

# Create the user-item matrix
user_item_matrix = np.zeros((num_users, num_items))
for row in data.itertuples():
    user_item_matrix[row.user_id - 1, row.item_id - 1] = row.rating

# Normalize ratings (subtract mean rating for each user)
mean_user_ratings = np.true_divide(user_item_matrix.sum(1), (user_item_matrix != 0).sum(1))
mean_user_ratings = np.nan_to_num(mean_user_ratings)  # Handle division by zero
normalized_matrix = user_item_matrix - mean_user_ratings[:, np.newaxis]
normalized_matrix[user_item_matrix == 0] = 0  # Keep zeros where there were no ratings

# Save the normalized matrix and mean ratings for later use in CUDA
np.savetxt("normalized_matrix.txt", normalized_matrix)
np.savetxt("mean_user_ratings.txt", mean_user_ratings)
print("Preprocessing complete. Files saved: normalized_matrix.txt, mean_user_ratings.txt")
