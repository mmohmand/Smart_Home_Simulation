import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from deap import base, creator, tools, algorithms
from sklearn.metrics import accuracy_score, precision_score

# Load the data
file_path = 'C:/Users/Murad Khan/ene_shr/Home_energy_data.csv'  # Update with your file path
data = pd.read_csv(file_path)

# Preprocess the data
data['DateTime'] = pd.to_datetime(data['DateTime'])
data = data.set_index('DateTime')

# Convert all columns to numeric, coercing errors
data = data.apply(pd.to_numeric, errors='coerce')

# Fill any missing values with 0
data.fillna(0, inplace=True)

# Convert energy data to kWh
# Conversion factor from watts to kilowatt-hours (kWh)
# 1 watt over 1 second = 1 / 1000 * 1 / 3600 kWh = 1 / 3,600,000 kWh
conversion_factor = 1 / 3600000

# Apply the conversion factor to the data
data = data * conversion_factor

# Aggregate to 15-minute intervals to reduce problem size
data = data.resample('15T').sum()

# Extract energy consumption data for appliances and solar generation
appliance_data = data.iloc[:, :-2]  # Exclude the 'Total_Energy' and 'Solar_Generation' columns
total_energy = data['Total_Energy']
solar_generation = data['Solar_Generation']

# Normalize appliance data
appliance_data_norm = appliance_data / appliance_data.max()

# Define the peak hours (for simplicity, let's assume peak hours are 6 PM to 10 PM)
peak_hours = (data.index.hour >= 18) & (data.index.hour < 22)

# Attention mechanism to weight the importance of different time intervals
def attention_weights(sequence):
    sequence = np.array(sequence)
    attention = np.exp(sequence) / np.sum(np.exp(sequence), axis=0)
    return attention

# Define the fitness function with attention mechanism
def fitness(individual):
    individual = np.array(individual).reshape(appliance_data_norm.shape)
    cost = np.where(peak_hours[:, None], 2, 1)
    attention = attention_weights(individual)
    runtime = np.sum(individual, axis=0)
    penalty = np.sum((runtime - 4) ** 2)  # Each appliance must run for 1 hour
    
    # Ensure all values are valid to prevent NaN
    cost = np.nan_to_num(cost)
    attention = np.nan_to_num(attention)
    individual = np.nan_to_num(individual)
    appliance_data_norm_values = np.nan_to_num(appliance_data_norm.values)
    
    total_cost = np.sum(cost * attention * individual * appliance_data_norm_values) + 1e8 * penalty
    return total_cost,

# Create the individual and population for GA
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=appliance_data_norm.size)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register the genetic operators
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)  # Increased mutation rate
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", fitness)

# Genetic Algorithm
def run_ga():
    random.seed(42)
    pop = toolbox.population(n=500)  # Increased population size
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields

    accuracy_over_time = []

    # Run the genetic algorithm
    for gen in range(200):  # Increased number of generations
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.7, mutpb=0.3)  # Adjusted crossover and mutation rates
        fits = map(toolbox.evaluate, offspring)

        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        pop = toolbox.select(offspring, k=len(pop))
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(offspring), **record)
        hof.update(pop)
        print(logbook.stream)

        # Calculate accuracy for this generation
        best_individual = hof[0]
        optimized_schedule = np.array(best_individual).reshape(appliance_data_norm.shape)
        y_true = (appliance_data.values > 0).astype(int)
        y_pred = optimized_schedule.astype(int)
        accuracy = accuracy_score(y_true.flatten(), y_pred.flatten())
        accuracy_over_time.append(accuracy)

    return hof, logbook, pop, accuracy_over_time

# Run the genetic algorithm
hof_ga, logbook_ga, final_pop_ga, accuracy_over_time_ga = run_ga()

# Extract the optimized schedule for GA
optimized_schedule_ga = np.array(hof_ga[0]).reshape(appliance_data_norm.shape)

# Create a new DataFrame for the optimized schedule for GA
optimized_data_ga = pd.DataFrame(optimized_schedule_ga * appliance_data_norm.values, index=appliance_data.index, columns=appliance_data.columns)

# Denormalize optimized data for GA
optimized_data_ga = optimized_data_ga * appliance_data.max()

# Calculate the total energy consumption for the optimized schedule
optimized_total_energy_ga = optimized_data_ga.sum(axis=1)

# Define parameters
B_max = 10  # Max battery capacity in kWh
P_sell = 0.16  # Selling price per kWh

# Initialize battery level
B = 0

# Calculate surplus energy, battery level, and revenue
S = np.maximum(solar_generation - optimized_total_energy_ga, 0)
E_saved = total_energy - optimized_total_energy_ga
S_total = S + E_saved

B_levels = []
S_grid = []
R_total = 0

for t in range(len(S_total)):
    if solar_generation[t] < optimized_total_energy_ga[t]:
        # Use battery if solar generation is less than demand
        energy_deficit = optimized_total_energy_ga[t] - solar_generation[t]
        if B >= energy_deficit:
            B -= energy_deficit
            energy_deficit = 0
        else:
            energy_deficit -= B
            B = 0
    else:
        energy_deficit = 0
    
    if B + S_total[t] - energy_deficit > B_max:
        S_grid_t = S_total[t] - energy_deficit - (B_max - B)
        B = B_max
    else:
        S_grid_t = 0
        B += S_total[t] - energy_deficit
    
    B_levels.append(B)
    S_grid.append(S_grid_t)
    R_total += S_grid_t * P_sell

# Convert lists to series for easier plotting
B_levels = pd.Series(B_levels, index=data.index)
S_grid = pd.Series(S_grid, index=data.index)
E_saved_series = pd.Series(E_saved, index=data.index)
solar_savings = pd.Series(S, index=data.index)

# Plot battery levels
plt.figure(figsize=(10, 6))
plt.plot(data.index, B_levels, label='Battery Level (kWh)')
plt.xlabel('Time')
plt.ylabel('Battery Level (kWh)')
plt.legend()
plt.title('Battery Levels Over Time')
plt.xticks(rotation=45)
plt.show()

# Plot grid surplus energy
plt.figure(figsize=(10, 6))
plt.plot(data.index, S_grid, label='Surplus Energy Sent to Grid (kWh)')
plt.xlabel('Time')
plt.ylabel('Surplus Energy (kWh)')
plt.legend()
plt.title('Surplus Energy Sent to Grid Over Time')
plt.xticks(rotation=45)
plt.show()

# Plot total revenue
plt.figure(figsize=(10, 6))
plt.plot(data.index, S_grid * P_sell, label='Revenue ($)')
plt.xlabel('Time')
plt.ylabel('Revenue ($)')
plt.legend()
plt.title('Revenue from Surplus Energy Over Time')
plt.xticks(rotation=45)
plt.show()

# Plot energy saved from scheduling
plt.figure(figsize=(10, 6))
plt.plot(data.index, E_saved_series, label='Energy Saved from Scheduling (kWh)')
plt.xlabel('Time')
plt.ylabel('Energy Saved (kWh)')
plt.legend()
plt.title('Energy Saved from Scheduling Over Time')
plt.xticks(rotation=45)
plt.show()

# Plot energy saved from solar generation
plt.figure(figsize=(10, 6))
plt.plot(data.index, solar_savings, label='Energy Saved from Solar Generation (kWh)')
plt.xlabel('Time')
plt.ylabel('Energy Saved (kWh)')
plt.legend()
plt.title('Energy Saved from Solar Generation Over Time')
plt.xticks(rotation=45)
plt.show()

print(f'Total Revenue from Selling Surplus Energy: ${R_total:.2f}')
