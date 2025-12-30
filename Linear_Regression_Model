import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

# The sample data
data = pd.read_csv('/content/sample_data/data.csv') 

# Compute correlation matrix
numeric_data = data.select_dtypes(include='number')
corr = numeric_data.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap")
plt.show() # Plot our data on a graph: x -> hours of sleep, y -> exam score
plt.scatter(data['Sleep_Hours'], data['Exam_Score']) 

# Define a function to perform one step of gradient descent
def gradient_descent(m_now, b_now, points, L):
  # Initialize the gradient for the slope (m) to zero
  m_gradient = 0

  # Initialize the gradient for the intercept (b) to zero
  b_gradient = 0

  # Get the total number of data points
  n = len(points)

  # Loop through each data point
  for i in range(n):
    # Extract the x-value (Sleep Hours) from the dataset
    x = points.iloc[i].Sleep_Hours

    # Extract the y-value (Exam Score) from the dataset
    y = points.iloc[i].Exam_Score

    # Accumulate the gradient of the loss function with respect to m
    m_gradient += -(2/n) * x * (y - (m_now * x + b_now))

    # Accumulate the gradient of the loss function with respect to b
    b_gradient += -(2/n) * (y - (m_now * x + b_now))

  # Update the slope (m) using the learning rate
  m = m_now - m_gradient * L

  # Update the intercept (b) using the learning rate
  b = b_now - b_gradient * L

  # Return the updated slope and intercept
  return m, b


# Initialize the slope (m) to 0
m = 0

# Initialize the intercept (b) to 0
b = 0

# Set the learning rate
L = 0.009

# Define the number of training iterations
epochs = 10000

# Run gradient descent for the specified number of epochs
for i in range(epochs):
  # Update m and b on each iteration using the gradient descent function
  m, b = gradient_descent(m, b, data, L)

# Print the final linear equation with values rounded to two decimals
print(f"y = {m.round(2)}x + {b.round(2)}")

# Plot the original data points as a scatter plot
plt.scatter(data.Sleep_Hours, data.Exam_Score, color='black')

# Plot the regression line using the learned m and b values
plt.plot(list(range(4, 10)), [m * x + b for x in range(4, 10)], color='red')

# Display the plot
plt.show()
