from policy import Policy
import numpy as np
class Policy2210xxx(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        # Student code here
        list_prods = observation["products"]
        sorted_prods = sorted(list_prods, key=lambda x: (x["size"][0] / x["size"][1], x["size"][0] * x["size"][1], x["quantity"]), reverse=True)
        for prod in sorted_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size
                    if stock_w < prod_w or stock_h < prod_h:
                        continue
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                return {"stock_idx": i, "size": prod_size, "position": (x, y)}
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}


class Policy2210xxx(Policy):
    def __init__(self):
        self.population_size = 50
        self.generations = 100
        self.mutation_rate = 0.01

    def get_action(self, observation, info):
        list_prods = observation["products"]

        # Initialize population
        population = [self.generate_individual(observation, idx) for idx in range(self.population_size)]

        for _ in range(self.generations):
            population = self.evolve_population(population, observation)

        # Get the best individual from the final population
        best_individual = max(population, key=lambda ind: self.evaluate_fitness(ind, observation))

        # Decode the best individual to get the action
        return self.decode_individual(best_individual, observation)

    def generate_individual(self, observation, idx):
        individual = []
        for j, prod in enumerate(observation["products"]):
            stock_idx = (j + idx) % len(observation["stocks"])
            stock = observation["stocks"][stock_idx]
            pos_x, pos_y = self.find_position(stock, prod["size"])
            if pos_x == -1 and pos_y == -1:
                continue  # Skip if no valid position is found
            individual.append((stock_idx, (pos_x, pos_y)))
        return individual

    def find_position(self, stock, prod_size):
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    return x, y
        return -1, -1

    def evolve_population(self, population, observation):
        # Selection
        selected_individuals = self.selection(population, observation)

        # Crossover
        offspring = []
        for i in range(0, len(population) // 2):
            parent1 = selected_individuals[i]
            parent2 = selected_individuals[len(selected_individuals) - 1 - i]
            child1, child2 = self.crossover(parent1, parent2)
            offspring.append(child1)
            offspring.append(child2)

        # Mutation
        for individual in offspring:
            self.mutate(individual, observation)

        # Combine and sort
        new_population = selected_individuals + offspring
        new_population = sorted(new_population, key=lambda ind: self.evaluate_fitness(ind, observation), reverse=True)
        return new_population[:self.population_size]

    def selection(self, population, observation):
        # Deterministic selection based on fitness
        sorted_population = sorted(population, key=lambda ind: self.evaluate_fitness(ind, observation), reverse=True)
        return sorted_population[:self.population_size // 2]

    def crossover(self, parent1, parent2):
        # Single point crossover
        crossover_point = len(parent1) // 2
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def mutate(self, individual, observation):
        for idx in range(len(individual)):
            if idx % (len(individual) // 2) == 0:
                stock_idx = (individual[idx][0] + 1) % len(observation["stocks"])
                stock = observation["stocks"][stock_idx]
                pos_x, pos_y = self.find_position(stock, observation["products"][idx]["size"])
                if pos_x == -1 and pos_y == -1:
                    continue  # Skip if no valid position is found
                individual[idx] = (stock_idx, (pos_x, pos_y))

    def evaluate_fitness(self, individual, observation):
        # Calculate fitness based on how well the products fit in the stocks
        fitness = 0
        for (stock_idx, (pos_x, pos_y)), prod in zip(individual, observation["products"]):
            if prod["quantity"] > 0:
                stock = observation["stocks"][stock_idx]
                if self._can_place_(stock, (pos_x, pos_y), prod["size"]):
                    fitness += 1  # Increment fitness for each valid placement
        return fitness

    def decode_individual(self, individual, observation):
        for (stock_idx, (pos_x, pos_y)), prod in zip(individual, observation["products"]):
            if prod["quantity"] > 0:
                if self._can_place_(observation["stocks"][stock_idx], (pos_x, pos_y), prod["size"]):
                    return {"stock_idx": stock_idx, "size": prod["size"], "position": (pos_x, pos_y)}
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
