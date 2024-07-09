import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

def calculate_percentile(data, percentile):
    """
    Calculate the given percentile of the data, ignoring NaN values.
    
    Parameters:
    data (pd.Series): The data series to calculate the percentile for.
    percentile (float): The percentile to calculate (between 0 and 100).
    
    Returns:
    float: The value at the given percentile.
    """
    # Remove NaN values
    cleaned_data = data.dropna()
    
    # Calculate the percentile
    return np.percentile(cleaned_data, percentile)

def plot_loan_duration_distribution(df):
    """
    Plot the loan duration distribution with a Gaussian overlay.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the loan data.
    """
    loan_durations = df['position_days'].dropna()
    
    # Define intervals extended up to 300 days
    bins = [0, 1, 3, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90] + list(range(100, 301, 10)) + [float('inf')]
    labels = ['0-1', '2-3', '3-7', '7-10', '10-15', '15-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90'] + [f'{i}-{i+9}' for i in range(90, 300, 10)] + ['300+']
    
    # Create a new 'interval' column with the intervals
    df['interval'] = pd.cut(loan_durations, bins=bins, labels=labels, right=False)
    
    # Calculate frequencies
    interval_counts = df['interval'].value_counts().sort_index()
    
    # Print the frequencies
    print(interval_counts)
    
    # Plot the histogram and Gaussian distribution
    plt.figure(figsize=(15, 8))

    # Histogram of the data
    count, bins, ignored = plt.hist(loan_durations, bins=30, density=True, alpha=0.6, color='g', edgecolor='black')

    # Calculate the mean and standard deviation
    mean = np.mean(loan_durations)
    std_dev = np.std(loan_durations)

    # Overlay the Gaussian distribution
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean, std_dev)

    plt.plot(x, p, 'k', linewidth=2)
    plt.xlabel('Loan Duration (days)')
    plt.ylabel('Density')
    plt.title('Loan Duration Distribution with Gaussian Overlay')

    plt.show()

def replace_nan_with_percentile(df, column, percentile_value):
    """
    Replace NaN values in the specified column with the given percentile value.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the loan data.
    column (str): The column in which to replace NaN values.
    percentile_value (float): The value to replace NaNs with.
    
    Returns:
    pd.DataFrame: The DataFrame with NaNs replaced.
    """
    df[column] = df[column].fillna(percentile_value)
    return print(df.head())

# Read the CSV file
df = pd.read_csv('2.RawLoans.csv')

# Calculate the 95th percentile
percentile_95 = calculate_percentile(df['position_days'], 95)
print(f"The 95th percentile of loan durations is: {percentile_95} days")

# Replace NaN values with the 95th percentile
#df = replace_nan_with_percentile(df, 'position_days', percentile_95)

# Plot the loan duration distribution
plot_loan_duration_distribution(df)

