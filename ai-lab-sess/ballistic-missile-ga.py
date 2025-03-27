import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import euclidean
import random
import copy
import time

class MissileTrajectoryGA:
    def __init__(self, 
                 start_position, 
                 target_position,
                 population_size=200,
                 generations=500,
                 crossover_rate=0.85,
                 mutation_rate=0.1,
                 tournament_size=5,
                 elitism_rate=0.1,
                 waypoints_count=10,
                 enemy_zones=None,
                 z_min=100,
                 z_max=10000,
                 max_acceleration=30,  # m/s^2, approximately 3G
                 target_tolerance=50   # meters
                ):
        # Basic parameters
        self.start_position = np.array(start_position)
        self.target_position = np.array(target_position)
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_rate = elitism_rate
        self.waypoints_count = waypoints_count
        self.z_min = z_min
        self.z_max = z_max
        self.max_acceleration = max_acceleration
        self.target_tolerance = target_tolerance
        
        # Set default enemy zones if none provided
        if enemy_zones is None:
            self.enemy_zones = [
                {"type": "sphere", "center": [5000, 5000, 3000], "radius": 2000},
                {"type": "cylinder", "center": [8000, 7000, 0], "radius": 1500, "height": 5000}
            ]
        else:
            self.enemy_zones = enemy_zones
            
        # Population and fitness tracking
        self.population = []
        self.fitness_history = []
        self.best_solution = None
        self.best_fitness = float('inf')
        
    def initialize_population(self):
        """Create initial population of random trajectories biased toward the target"""
        self.population = []
        
        for _ in range(self.population_size):
            # Create a trajectory with waypoints_count intermediate points
            trajectory = [self.start_position.copy()]
            
            # Generate intermediate waypoints biased toward target
            for i in range(self.waypoints_count):
                # Bias factor: 0 at start, 1 at end
                bias = (i + 1) / (self.waypoints_count + 1)
                
                # Biased midpoint + random variation
                midpoint = self.start_position + bias * (self.target_position - self.start_position)
                
                # Add random variation that decreases as we approach the target
                variation = (1 - bias) * 0.5 * np.linalg.norm(self.target_position - self.start_position)
                random_offset = np.random.uniform(-variation, variation, 3)
                
                # Ensure z-coordinate is within bounds
                waypoint = midpoint + random_offset
                waypoint[2] = max(min(waypoint[2], self.z_max), self.z_min)
                
                trajectory.append(waypoint)
            
            # Add target as final point
            trajectory.append(self.target_position.copy())
            
            self.population.append(np.array(trajectory))
            
    def calculate_fuel_cost(self, trajectory):
        """Calculate fuel consumption based on distance and altitude"""
        total_cost = 0
        turning_cost = 0
        
        for i in range(len(trajectory) - 1):
            current_point = trajectory[i]
            next_point = trajectory[i+1]
            
            # Distance between waypoints
            distance = np.linalg.norm(next_point - current_point)
            
            # Altitude-based fuel cost factor (higher altitude = higher base fuel cost)
            # But higher altitude has less drag, so it's a trade-off
            altitude = (current_point[2] + next_point[2]) / 2
            altitude_factor = 0.7 + (altitude / self.z_max) * 0.6
            
            # Calculate segment fuel cost
            segment_cost = distance * altitude_factor
            
            # Add turning cost if not the first segment
            if i > 0:
                prev_vector = trajectory[i] - trajectory[i-1]
                current_vector = trajectory[i+1] - trajectory[i]
                
                # Normalize vectors
                prev_vector = prev_vector / np.linalg.norm(prev_vector)
                current_vector = current_vector / np.linalg.norm(current_vector)
                
                # Calculate angle between vectors (dot product)
                cos_angle = np.clip(np.dot(prev_vector, current_vector), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                
                # Higher turning_cost for sharper turns
                turning_penalty = angle**2 * 100
                turning_cost += turning_penalty
            
            total_cost += segment_cost
            
        # Add turning cost with penalty factor
        alpha = 0.2  # Penalty factor for turns
        return total_cost + alpha * turning_cost
    
    def check_enemy_zone_violation(self, trajectory):
        """Check if trajectory violates enemy defense zones"""
        violation_count = 0
        violation_distance = 0
        
        for point in trajectory:
            for zone in self.enemy_zones:
                if zone["type"] == "sphere":
                    # Check if point is inside spherical zone
                    distance = np.linalg.norm(point - np.array(zone["center"]))
                    if distance < zone["radius"]:
                        violation_count += 1
                        violation_distance += zone["radius"] - distance
                        
                elif zone["type"] == "cylinder":
                    # Check if point is inside cylindrical zone
                    center = np.array(zone["center"])
                    # For cylinder, we check x-y distance and z height separately
                    horizontal_distance = np.linalg.norm(point[:2] - center[:2])
                    if (horizontal_distance < zone["radius"] and 
                        0 <= point[2] <= zone["height"] + center[2]):
                        violation_count += 1
                        violation_distance += zone["radius"] - horizontal_distance
        
        return violation_count, violation_distance
    
    def check_acceleration_violation(self, trajectory):
        """Check if trajectory violates maximum acceleration constraint"""
        violation_count = 0
        violation_magnitude = 0
        
        # Assume constant time between waypoints for simplicity
        time_step = 1.0  # can be adjusted based on mission parameters
        
        # Need at least 3 points to calculate acceleration
        if len(trajectory) < 3:
            return 0, 0
        
        # Calculate velocities between consecutive points
        velocities = []
        for i in range(len(trajectory) - 1):
            displacement = trajectory[i+1] - trajectory[i]
            velocity = displacement / time_step
            velocities.append(velocity)
        
        # Calculate accelerations between consecutive velocities
        for i in range(len(velocities) - 1):
            delta_v = velocities[i+1] - velocities[i]
            acceleration = np.linalg.norm(delta_v) / time_step
            
            if acceleration > self.max_acceleration:
                violation_count += 1
                violation_magnitude += acceleration - self.max_acceleration
                
        return violation_count, violation_magnitude
    
    def check_altitude_violation(self, trajectory):
        """Check if trajectory violates altitude constraints"""
        violation_count = 0
        violation_magnitude = 0
        
        for point in trajectory:
            altitude = point[2]
            
            if altitude < self.z_min:
                violation_count += 1
                violation_magnitude += self.z_min - altitude
            elif altitude > self.z_max:
                violation_count += 1
                violation_magnitude += altitude - self.z_max
                
        return violation_count, violation_magnitude
    
    def check_target_hit(self, trajectory):
        """Check if the trajectory hits the target within tolerance"""
        final_position = trajectory[-1]
        distance_to_target = np.linalg.norm(final_position - self.target_position)
        
        if distance_to_target > self.target_tolerance:
            return False, distance_to_target
        return True, distance_to_target
    
    def calculate_fitness(self, trajectory):
        """Calculate fitness value for a trajectory"""
        # Base fitness is fuel cost
        fuel_cost = self.calculate_fuel_cost(trajectory)
        
        # Calculate constraint violations
        enemy_violations, enemy_distance = self.check_enemy_zone_violation(trajectory)
        accel_violations, accel_magnitude = self.check_acceleration_violation(trajectory)
        alt_violations, alt_magnitude = self.check_altitude_violation(trajectory)
        target_hit, target_distance = self.check_target_hit(trajectory)
        
        # Penalty factors
        lambda_enemy = 1000
        lambda_accel = 500
        lambda_alt = 800
        lambda_target = 2000 if not target_hit else 0
        
        # Calculate penalties
        enemy_penalty = lambda_enemy * enemy_violations + 10 * enemy_distance
        accel_penalty = lambda_accel * accel_violations + 5 * accel_magnitude
        alt_penalty = lambda_alt * alt_violations + 5 * alt_magnitude
        target_penalty = lambda_target * target_distance
        
        # Total fitness (lower is better)
        total_fitness = fuel_cost + enemy_penalty + accel_penalty + alt_penalty + target_penalty
        
        return total_fitness
    
    def tournament_selection(self):
        """Select parent using tournament selection"""
        # Randomly select tournament_size individuals
        tournament = random.sample(range(len(self.population)), self.tournament_size)
        
        # Select the best one from the tournament
        best_idx = tournament[0]
        best_fitness = self.calculate_fitness(self.population[best_idx])
        
        for idx in tournament[1:]:
            fitness = self.calculate_fitness(self.population[idx])
            if fitness < best_fitness:
                best_idx = idx
                best_fitness = fitness
                
        return self.population[best_idx].copy()
    
    def crossover(self, parent1, parent2):
        """Perform crossover between two parent trajectories"""
        if random.random() > self.crossover_rate:
            return parent1.copy()
        
        # Ensure both parents have the same number of waypoints
        assert len(parent1) == len(parent2)
        
        # Two-point crossover
        points = sorted(random.sample(range(1, len(parent1) - 1), 2))
        
        # Create child by combining parts from both parents
        child = np.vstack([
            parent1[:points[0]],
            parent2[points[0]:points[1]],
            parent1[points[1]:]
        ])
        
        return child
    
    def mutate(self, trajectory):
        """Apply mutation to a trajectory"""
        mutated = trajectory.copy()
        
        # Keep first and last points fixed (start and target)
        for i in range(1, len(mutated) - 1):
            # Apply mutation with probability mutation_rate
            if random.random() < self.mutation_rate:
                # Get neighboring points for reference
                prev_point = mutated[i-1]
                next_point = mutated[i+1]
                
                # Calculate midpoint
                midpoint = (prev_point + next_point) / 2
                
                # Calculate maximum allowed deviation
                max_deviation = np.linalg.norm(next_point - prev_point) * 0.5
                
                # Apply random mutation
                random_offset = np.random.uniform(-max_deviation, max_deviation, 3)
                
                # Create new point
                new_point = midpoint + random_offset
                
                # Ensure z-coordinate is within bounds
                new_point[2] = max(min(new_point[2], self.z_max), self.z_min)
                
                mutated[i] = new_point
                
        return mutated
    
    def evolve(self):
        """Run the genetic algorithm for the specified number of generations"""
        # Initialize population
        self.initialize_population()
        
        # Track fitness history
        self.fitness_history = []
        
        # Evolve for specified number of generations
        for generation in range(self.generations):
            # Calculate fitness for all individuals
            fitness_values = [self.calculate_fitness(ind) for ind in self.population]
            
            # Record best fitness
            best_idx = np.argmin(fitness_values)
            best_gen_fitness = fitness_values[best_idx]
            self.fitness_history.append(best_gen_fitness)
            
            # Update best overall solution
            if best_gen_fitness < self.best_fitness:
                self.best_fitness = best_gen_fitness
                self.best_solution = self.population[best_idx].copy()
                
            # Optional: Print progress
            if generation % 50 == 0:
                print(f"Generation {generation}: Best Fitness = {best_gen_fitness:.2f}")
                
            # Create new population using elitism, crossover, and mutation
            
            # Sort population by fitness (ascending: lower is better)
            sorted_indices = np.argsort(fitness_values)
            
            # Apply elitism - keep best individuals
            elite_count = int(self.population_size * self.elitism_rate)
            new_population = [self.population[idx].copy() for idx in sorted_indices[:elite_count]]
            
            # Fill the rest with crossover and mutation
            while len(new_population) < self.population_size:
                # Select parents
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                
                # Create child through crossover
                child = self.crossover(parent1, parent2)
                
                # Apply mutation
                child = self.mutate(child)
                
                new_population.append(child)
                
            # Replace old population
            self.population = new_population
            
        # Set best solution if not found
        if self.best_solution is None:
            best_idx = np.argmin([self.calculate_fitness(ind) for ind in self.population])
            self.best_solution = self.population[best_idx].copy()
            self.best_fitness = self.calculate_fitness(self.best_solution)
            
        print(f"Final Best Fitness: {self.best_fitness:.2f}")
        
    def plot_result(self):
        """Plot the best trajectory in 3D space along with enemy zones"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot best trajectory
        trajectory = self.best_solution
        x = trajectory[:, 0]
        y = trajectory[:, 1]
        z = trajectory[:, 2]
        
        ax.plot(x, y, z, 'b-', linewidth=2, label='Missile Trajectory')
        ax.scatter(x[0], y[0], z[0], c='g', s=100, label='Launch Point')
        ax.scatter(x[-1], y[-1], z[-1], c='r', s=100, label='Target')
        
        # Plot waypoints
        ax.scatter(x[1:-1], y[1:-1], z[1:-1], c='b', s=30)
        
        # Plot enemy zones
        for zone in self.enemy_zones:
            if zone["type"] == "sphere":
                # Plot sphere
                center = zone["center"]
                radius = zone["radius"]
                
                # Create sphere
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                x_sphere = center[0] + radius * np.cos(u) * np.sin(v)
                y_sphere = center[1] + radius * np.sin(u) * np.sin(v)
                z_sphere = center[2] + radius * np.cos(v)
                
                ax.plot_surface(x_sphere, y_sphere, z_sphere, color='r', alpha=0.2)
                
            elif zone["type"] == "cylinder":
                # Plot cylinder
                center = zone["center"]
                radius = zone["radius"]
                height = zone["height"]
                
                # Create cylinder
                theta = np.linspace(0, 2*np.pi, 30)
                z_cyl = np.linspace(center[2], center[2] + height, 10)
                theta_grid, z_grid = np.meshgrid(theta, z_cyl)
                
                x_cyl = center[0] + radius * np.cos(theta_grid)
                y_cyl = center[1] + radius * np.sin(theta_grid)
                
                ax.plot_surface(x_cyl, y_cyl, z_grid, color='r', alpha=0.2)
        
        # Set labels and title
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Altitude (m)')
        ax.set_title('Optimized Missile Trajectory')
        
        # Add legend
        ax.legend()
        
        # Plot fitness history
        plt.figure(figsize=(10, 6))
        plt.plot(self.fitness_history)
        plt.xlabel('Generation')
        plt.ylabel('Fitness (lower is better)')
        plt.title('Fitness History')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def analyze_solution(self):
        """Analyze the best solution and print detailed information"""
        if self.best_solution is None:
            print("No solution found. Run evolve() first.")
            return
        
        trajectory = self.best_solution
        
        # Calculate metrics
        fuel_cost = self.calculate_fuel_cost(trajectory)
        enemy_violations, enemy_distance = self.check_enemy_zone_violation(trajectory)
        accel_violations, accel_magnitude = self.check_acceleration_violation(trajectory)
        alt_violations, alt_magnitude = self.check_altitude_violation(trajectory)
        target_hit, target_distance = self.check_target_hit(trajectory)
        
        # Print report
        print("=== Trajectory Analysis ===")
        print(f"Total waypoints: {len(trajectory)}")
        print(f"Total distance: {sum(np.linalg.norm(trajectory[i+1] - trajectory[i]) for i in range(len(trajectory)-1)):.2f} m")
        print(f"Fuel cost: {fuel_cost:.2f}")
        print("\n=== Constraint Violations ===")
        print(f"Enemy zone violations: {enemy_violations}")
        print(f"Acceleration constraint violations: {accel_violations}")
        print(f"Altitude constraint violations: {alt_violations}")
        print(f"Target hit within tolerance: {'Yes' if target_hit else 'No'}")
        print(f"Distance to target: {target_distance:.2f} m")
        
        return {
            "fuel_cost": fuel_cost,
            "enemy_violations": enemy_violations,
            "accel_violations": accel_violations,
            "alt_violations": alt_violations,
            "target_hit": target_hit,
            "target_distance": target_distance
        }


# Example usage
def main():


    start_position = [0, 0, 500]  # Starting coordinates (x, y, z) in meters
    target_position = [15000, 15000, 300]  # Target coordinates in meters
    
    # Enemy defense zones (examples)
    enemy_zones = [
        {"type": "sphere", "center": [5000, 5000, 3000], "radius": 2000},
        {"type": "sphere", "center": [10000, 8000, 2000], "radius": 2500},
        {"type": "cylinder", "center": [8000, 12000, 0], "radius": 1800, "height": 5000}
    ]
    
    # Create optimizer with custom parameters
    missile_ga = MissileTrajectoryGA(
        start_position=start_position,
        target_position=target_position,
        population_size=200,
        generations=300,
        crossover_rate=0.85,
        mutation_rate=0.15,
        tournament_size=5,
        elitism_rate=0.1,
        waypoints_count=8,
        enemy_zones=enemy_zones,
        z_min=100,
        z_max=8000,
        max_acceleration=30,
        target_tolerance=50
    )
    
    # Run the optimization
    start_time = time.time()
    missile_ga.evolve()
    end_time = time.time()
    
    print(f"Optimization completed in {end_time - start_time:.2f} seconds")
    
    # Analyze and visualize results
    missile_ga.analyze_solution()
    missile_ga.plot_result()

















# this is used to run the script....

if __name__ == "__main__":
    main()