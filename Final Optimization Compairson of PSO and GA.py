import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from deap import base, creator, tools, algorithms
from sklearn.metrics import accuracy_score, precision_score
from scipy.optimize import differential_evolution
from pyswarm import pso

# Load the data
file_path = 'C:/Users/Murad Khan/ene_shr/Home_energy_data.csv'  # Update with your file path
data = pd.read_csv(file_path)

# Preprocess the data
data['DateTime'] = pd.to_datetime(data['DateTime'])
data = data.set_index('DateTime')

# Aggregate to 15-minute intervals to reduce problem size
data = data.resample('15T').sum()

# Extract energy consumption data for appliances
appliance_data = data.iloc[:, :-1]  # Exclude the 'Total_Energy' column
total_energy = data['Total_Energy']

# Normalize appliance data
appliance_data_norm = appliance_data / appliance_data.max()

# Define the peak hours (for simplicity, let's assume peak hours are 6 PM to 10 PM)
peak_hours = (data.index.hour >= 18) & (data.index.hour < 22)

# Attention mechanism to weight the importance of different time intervals
def attention_weights(sequence):
    attention = np.exp(sequence) / np.sum(np.exp(sequence), axis=0)
    return attention

# Define the fitness function with attention mechanism
def fitness(individual):
    individual = np.array(individual).reshape(appliance_data_norm.shape)
    cost = np.where(peak_hours[:, None], 2, 1)
    attention = attention_weights(individual)
    runtime = np.sum(individual, axis=0)
    penalty = np.sum((runtime - 4) ** 2)  # Each appliance must run for 1 hour
    total_cost = np.sum(cost * attention * individual * appliance_data_norm.values) + 1e8 * penalty
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

# Particle Swarm Optimization (PSO)
def pso_fitness(individual):
    individual = individual.reshape(appliance_data_norm.shape)
    cost = np.where(peak_hours[:, None], 2, 1)
    attention = attention_weights(individual)
    runtime = np.sum(individual, axis=0)
    penalty = np.sum((runtime - 4) ** 2)  # Each appliance must run for 1 hour
    total_cost = np.sum(cost * attention * individual * appliance_data_norm.values) + 1e8 * penalty
    return total_cost

def run_pso():
    lower_bounds = np.zeros(appliance_data_norm.size)
    upper_bounds = np.ones(appliance_data_norm.size)

    best_position, best_cost = pso(pso_fitness, lower_bounds, upper_bounds, swarmsize=500, maxiter=200)
    
    optimized_schedule = best_position.reshape(appliance_data_norm.shape)
    y_true = (appliance_data.values > 0).astype(int)
    y_pred = optimized_schedule.astype(int)
    accuracy = accuracy_score(y_true.flatten(), y_pred.flatten())
    
    return optimized_schedule, accuracy

# Ant Colony Optimization (ACO) placeholder
# ACO is quite complex and does not have a standard library for this specific type of problem,
# thus this placeholder is to show how one might structure the solution if using a library like 'acopy'.
def run_aco():
    # Placeholder for ACO implementation
    # For a real ACO implementation, you would need a library or custom implementation
    pass

# Run the optimization algorithms
hof_ga, logbook_ga, final_pop_ga, accuracy_over_time_ga = run_ga()
optimized_schedule_pso, accuracy_pso = run_pso()
# optimized_schedule_aco, accuracy_aco = run_aco()  # Uncomment if implementing ACO

# Extract the optimized schedule for GA
optimized_schedule_ga = np.array(hof_ga[0]).reshape(appliance_data_norm.shape)

# Create a new DataFrame for the optimized schedule for GA and PSO
optimized_data_ga = pd.DataFrame(optimized_schedule_ga * appliance_data_norm.values, index=appliance_data.index, columns=appliance_data.columns)
optimized_data_pso = pd.DataFrame(optimized_schedule_pso * appliance_data_norm.values, index=appliance_data.index, columns=appliance_data.columns)

# Denormalize optimized data for GA and PSO
optimized_data_ga = optimized_data_ga * appliance_data.max()
optimized_data_pso = optimized_data_pso * appliance_data.max()

# Calculate the total energy consumption for the optimized schedule
optimized_total_energy_ga = optimized_data_ga.sum(axis=1)
optimized_total_energy_pso = optimized_data_pso.sum(axis=1)

# Plot the difference between actual and optimized total energy consumption for 24 hours
plt.figure(figsize=(10, 6))
plt.plot(data.index[-96:], total_energy[-96:], label='Actual Total Energy Consumption')
plt.plot(data.index[-96:], optimized_total_energy_ga[-96:], label='Optimized Total Energy Consumption (GA)')
plt.plot(data.index[-96:], optimized_total_energy_pso[-96:], label='Optimized Total Energy Consumption (PSO)')
plt.xlabel('Time')
plt.ylabel('Total Energy Consumption')
plt.legend()
plt.title('Total Energy Consumption Before and After Optimization (Last 24 Hours)')
plt.xticks(rotation=45)
plt.show()

# Plot energy consumption of each appliance in separate plots for GA
appliance_columns = appliance_data.columns
fig, axes = plt.subplots(len(appliance_columns), 1, figsize=(15, 2*len(appliance_columns)), sharex=True)
for i, col in enumerate(appliance_columns):
    axes[i].plot(data.index[-96:], appliance_data[col][-96:], label=f'Actual {col}')
    axes[i].plot(data.index[-96:], optimized_data_ga[col][-96:], label=f'Optimized {col} (GA)')
    axes[i].set_ylabel('Watt')
    axes[i].legend()
axes[-1].set_xlabel('Time')
plt.tight_layout()
plt.show()

# Plot energy consumption of each appliance in separate plots for PSO
fig, axes = plt.subplots(len(appliance_columns), 1, figsize=(15, 2*len(appliance_columns)), sharex=True)
for i, col in enumerate(appliance_columns):
    axes[i].plot(data.index[-96:], appliance_data[col][-96:], label=f'Actual {col}')
    axes[i].plot(data.index[-96:], optimized_data_pso[col][-96:], label=f'Optimized {col} (PSO)')
    axes[i].set_ylabel('Watt')
    axes[i].legend()
axes[-1].set_xlabel('Time')
plt.tight_layout()
plt.show()

# Additional plots for genetic algorithm performance

# Plotting minimum and average fitness values over generations
gen = logbook_ga.select("gen")
min_fitness_values = logbook_ga.select("min")
avg_fitness_values = logbook_ga.select("avg")

plt.figure(figsize=(10, 6))
plt.plot(gen, min_fitness_values, label='Minimum Fitness (GA)')
plt.plot(gen, avg_fitness_values, label='Average Fitness (GA)')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.title('Genetic Algorithm Performance')
plt.show()

# Box plots of fitness values in the population for each generation (GA)
fitness_values = [ind.fitness.values[0] for ind in final_pop_ga]
plt.figure(figsize=(10, 6))
plt.boxplot(fitness_values)
plt.xlabel('Generation')
plt.ylabel('Fitness Values')
plt.title('Fitness Values Distribution in Final Population (GA)')
plt.show()

# Accuracy and precision for GA
y_true = (appliance_data.values > 0).astype(int)
y_pred_ga = optimized_schedule_ga.astype(int)
accuracy_ga = accuracy_score(y_true.flatten(), y_pred_ga.flatten())
precision_ga = precision_score(y_true.flatten(), y_pred_ga.flatten())

print(f'Accuracy (GA): {accuracy_ga:.2f}')
print(f'Precision (GA): {precision_ga:.2f}')

# Plot accuracy over generations for GA
plt.figure(figsize=(10, 6))
plt.plot(range(200), accuracy_over_time_ga, label='Accuracy (GA)')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Generations (GA)')
plt.show()

# Print accuracy for PSO
print(f'Accuracy (PSO): {accuracy_pso:.2f}')

# For ACO, you would follow similar steps to extract, plot, and calculate the metrics.

# Note: This code assumes the ACO implementation would be similar to the GA and PSO.
# Since there is no standard ACO library for this problem, it is left as a placeholder.
