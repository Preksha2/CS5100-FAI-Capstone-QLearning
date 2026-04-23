# CS5100 FAI Capstone - Q-Learning Convergence Study

CS5100 Foundations of Artificial Intelligence, Spring 2026
Instructor: Jonathan Mwaura
Author: Preksha Morbia (NUID: 002523931)

## Paper
Watkins & Dayan (1992). Q-learning. Machine Learning, 8(3-4), 279-292.

## What This Does
Implements and verifies Q-learning convergence, then compares Q-learning, SARSA, and Double Q-learning across FrozenLake 4x4, FrozenLake 8x8, CliffWalking, and Taxi environments.

## How to Run
pip install numpy matplotlib
python3 main.py

Takes about 3-4 minutes. Generates results in results/ and figures in figures/.

## Files
- environments.py - Four RL environments
- algorithms.py - Value iteration, Q-learning, SARSA, Double Q-learning
- run_experiments.py - Runs all experiments
- visualizations.py - Generates figures
- main.py - Entry point
