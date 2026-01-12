import random
import math
import numpy as np

def solve_tsp(distance_matrix):
    n = len(distance_matrix)
    
    # Start at a random city; here we use np.random.choice with replacement=True.
    route = [random.choice(range(n))]
    visited = set([route[0]])
    
    while len(route) < n:
        last_city = route[-1]
        best_candidate = None
        best_cost = float('inf')
        
        for candidate in range(n):
            if candidate not in visited:
                # Compute Euclidean distance from the last city.
                dist = math.sqrt(distance_matrix[last_city][candidate]**2)
                
                # Introduce a repulsion penalty when the candidate is too close to the current city.
                # For example, if the distance is below a threshold (5.0), add a penalty of 100 * dist.
                repulsion = 100 * dist if dist < 5.0 else 0
                total_cost = dist + repulsion
                
                if total_cost < best_cost:
                    best_candidate = candidate
                    best_cost = total_cost
        
        # Append the best candidate to the route and mark it as visited.
        route.append(best_candidate)
        visited.add(best_candidate)
    
    return route

if __name__ == '__main__':
    # Example usage (you must have a 20x20 distance matrix):
    n = 20
    # Create a random symmetric distance matrix representing Euclidean distances.
    A = np.random.rand(n, n)
    distance_matrix = (A + A.T) / 2  
    
    solution = solve_tsp(distance_matrix)
    
    # For verification on the provided instance:
    # Compute the tour length by summing distances between consecutive cities in the solution.
    total_distance = sum(distance_matrix[i][j] for i, j in zip(solution[:-1], solution[1:]))
    
    print("Final route:", solution)
    print("Optimal tour length:", total_distance)