# 2048 AI with Deep Q-Learning
This project implements a reinforcement learning AI that learns to play the classic 2048 game using Deep Q-Learning with PyTorch. The AI agent progressively learns optimal strategies through self-play and can achieve high scores with sufficient training.

## Features
* Complete 2048 game implementation with Pygame visualization
* Deep Q-Network (DQN) architecture with convolutional layers for state processing
* Advanced reward shaping to encourage effective strategy development
* Support for the "snake pattern" strategy (a proven effective approach for 2048)
* Training progress visualization and model checkpointing
* Evaluation mode to showcase the trained agent's performance

## Installation
### Requirements
This project requires the following dependencies:
`torch       # Neural network framework`
`numpy       # Array operations and game logic`
`matplotlib  # Training visualization`
`pygame      # Game visualization`

### Setup
Install all required dependencies with:
`pip install torch numpy matplotlib pygame`
