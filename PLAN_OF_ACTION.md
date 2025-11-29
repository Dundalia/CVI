# Project Status Update: Tabular CVI Validation

## 1. Implemented Algorithms
We have successfully implemented and validated the following algorithms in a tabular setting (FrozenLake-v1 8x8 Slippery and Taxi-v3):
- **Policy Iteration (PI)**: Classical baseline.
- **Value Iteration (VI)**: Classical baseline.
- **Categorical Distributional RL (C51)**: Distributional baseline using fixed atoms.
- **Characteristic Value Iteration (CVI)**: Our novel frequency-domain approach.

## 2. Validation Metrics
To confirm the correctness of the CVI implementation, we compared it against the baselines using the following metrics:

- **Expected Initial State Value**: 
  The expected return calculated from the learned value function, averaged over 1000 sampled initial states. This checks if the algorithm's internal estimate matches the environment's start distribution.

- **MC Avg Return**: 
  The average return obtained from 1000 Monte Carlo rollouts using the learned policy. This serves as the "ground truth" performance metric to verify that the value function actually converged to an optimal policy.

- **Mean V Value**: 
  The average of the value function over the entire state space, tracked over iterations. This visualizes the convergence speed and stability.

- **Final Mean V Value**: 
  The final snapshot of the Mean V Value. This provides a single scalar for quick comparison across algorithms that may converge at different rates.

## 3. Distributional Validation (CDF Comparison)
Beyond scalar values, we validated CVI as a *distributional* algorithm by comparing the learned return distribution against:
1.  **Monte Carlo Ground Truth**: The empirical distribution of returns from actual rollouts.
2.  **C51 Baseline**: A proven distributional algorithm using categorical atoms.

### Why CDF?
We chose to plot the **Cumulative Distribution Function (CDF)** rather than the Probability Density Function (PDF) for three reasons:

1.  **Discrete vs. Continuous Mismatch**: FrozenLake returns are discrete (spikes), while CVI produces a continuous approximation (hills). A PDF plot shows little overlap, whereas a CDF plot clearly shows the CVI distribution "hugging" the steps of the discrete ground truth.
2.  **Artifact Suppression**: The Inverse FFT used to reconstruct the distribution from CVI's characteristic function introduces high-frequency "ringing" (Gibbs phenomenon) in the PDF. Integration into the CDF smooths these artifacts out, providing a cleaner view of the distribution's shape.
3.  **Risk Assessment**: The CDF allows for direct visual comparison of "stochastic dominance" and tail risks (the intercept at y=0), which are key properties in distributional RL.

**Result**: The CVI CDF closely matches both the Monte Carlo ground truth and the C51 baseline, confirming that the algorithm correctly learns the full distribution of returns, not just the expectation.

## 4. Next Steps
- **Transition to Function Approximation**: Move beyond tabular settings to Deep RL.
- **New Environment**: Test on environments with continuous state spaces (CartPole, Acrobot or even directly go to Atari).
- **Deep CVI Implementation**:
    - Design a Neural Network to output Characteristic Functions (Real/Imaginary parts).
    - Implement the frequency-domain L2 loss function.
    - Train and make it stable, might need to implement techniques like target networks and experience replay.
