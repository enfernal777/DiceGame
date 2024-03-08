import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def dice_pmf(n_sides, n_dice):
    """
    Calculates the PMF of the sum of n_dice dice rolls with n_sides sides.
    """
    # Generate all possible rolls
    all_rolls = np.array(np.meshgrid(*[range(1, n_sides + 1)] * n_dice)).T.reshape(-1, n_dice)
    
    # Calculate the sum of each roll
    roll_sums = all_rolls.sum(axis=1)
    
    # Calculate the PMF
    pmf_values, pmf_counts = np.unique(roll_sums, return_counts=True)
    pmf = pmf_counts / len(roll_sums)
    
    return pmf_values, pmf

def plot_dice_distribution(n_dice):
    """
    Plots the normal distribution of the sum of n_dice dice rolls.
    """
    # Calculate the PMF
    pmf_values, pmf = dice_pmf(6, n_dice)
    
    # Calculate the mean and standard deviation
    mean = np.mean(pmf_values)
    std = np.std(pmf_values, ddof=1)
    
    # Generate x values for the normal distribution curve
    x = np.linspace(pmf_values[0], pmf_values[-1], 1000)
    
    # Generate the normal distribution curve
    y = norm.pdf(x, mean, std)
    
    # Plot the PMF and normal distribution curve
    plt.bar(pmf_values, pmf, width=0.8, align='center')
    plt.plot(x, y, linewidth=2, color='r')
    
    # Add labels and title
    plt.xlabel('Sum of dice rolls')
    plt.ylabel('Probability')
    plt.title(f'Distribution of {n_dice} dice rolls')
    plt.show()

# Prompt the user for the number of dice
n_dice = int(input('Enter the number of dice: '))

# Generate the normal distribution graph
plot_dice_distribution(n_dice)
