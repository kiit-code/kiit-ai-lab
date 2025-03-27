# Missile Trajectory Optimization using Genetic Algorithm


## 1. Encoding Scheme (Trajectory Representation)

The most effective encoding scheme for missile trajectories is a direct waypoint representation:

- **Structure**: A sequence of 3D coordinates (x, y, z) representing points the missile passes through
- **Fixed endpoints**: The start (launch position) and end (target) points remain fixed
- **Variable waypoints**: 8-12 intermediate waypoints that can be optimized
- **Benefits**:
  - Intuitive representation that directly maps to physical space
  - Easily visualized and interpreted
  - Allows natural implementation of physical constraints
  - Supports efficient genetic operators

This approach is superior to parametric curves or angle-based representations because it directly handles spatial constraints and provides granular control over the trajectory.

## 2. Fitness Function (Balancing Objectives and Penalties)

The fitness function combines fuel consumption minimization with constraint penalties:

```
Fitness = FuelCost + λ₁·EnemyZonePenalty + λ₂·AccelerationPenalty + λ₃·AltitudePenalty + λ₄·TargetPenalty
```

Where:
- **Fuel Cost**: Distance-based with altitude factors (higher altitude = higher base cost but less drag)
- **Enemy Zone Penalty**: Both violation count and penetration depth
- **Acceleration Penalty**: Based on exceeding G-force limits between consecutive segments
- **Altitude Penalty**: For violating minimum/maximum altitude constraints
- **Target Penalty**: Distance-based penalty for missing the target beyond tolerance

The penalty coefficients (λ values) should be dynamically adjusted during evolution, starting high to enforce feasibility then gradually reducing to allow optimization.

## 3. Genetic Operators (Selection, Crossover, Mutation)

### Selection
- **Tournament Selection** (tournament size 5-7)
  - Selects parent by sampling random candidates and picking the best
  - Provides good selection pressure while preserving diversity
  - More robust than roulette wheel selection for this application

### Crossover
- **Two-Point Crossover** with 0.85 probability
  - Exchanges trajectory segments between parents
  - Maintains start and end points
  - Creates viable offspring that blend parental characteristics

### Mutation
- **Variable-Range Gaussian Mutation** with 0.1-0.15 probability
  - Adds random perturbations to waypoints with magnitude proportional to segment length
  - Biased toward maintaining trajectory smoothness
  - Mutation range decreases as generations progress
  - Special handling for altitude dimension to maintain flight envelope

## 4. Constraint Handling

The most effective approach uses a combination of:

- **Penalty Functions**: For constraints that can be partially violated
  - Enemy zones with weighted distance-based penalties
  - Acceleration constraints with magnitude-based penalties
  
- **Repair Methods**: For critical constraints
  - Altitude bounds enforced during mutation
  - Target position preserved in genetic operations

This hybrid approach ensures feasible solutions while allowing exploration of the solution space near constraint boundaries.

## 5. Termination Criteria

The algorithm should terminate based on:

1. **Maximum generations**: 300-500 generations as a hard limit
2. **Convergence detection**: No improvement in best fitness for 50 generations
3. **Feasibility achievement**: All critical constraints satisfied with fitness below threshold
4. **Time limit**: Optional computational budget constraint

## Additional Considerations

- **Population size**: 200-300 individuals provides good diversity without excessive computation
- **Elitism rate**: 10% preservation of best solutions ensures convergence
- **Initialization strategy**: Biased random initialization toward straight-line path improves starting solutions
- **Parallel evaluation**: Fitness calculation can be parallelized for efficiency
- **Adaptive parameters**: Mutation rate can decrease over generations to fine-tune solutions

