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
```
torch       # Neural network framework
numpy       # Array operations and game logic
matplotlib  # Training visualization
pygame      # Game visualization
```

### Setup
Install all required dependencies with:

```
pip install torch numpy matplotlib pygame
```

## Usage
### Training
```
python 2048_ai.py
```

This will:
0. Initialize the 2048 game environment
0. Create and train a DQN agent for 50,000 episodes
0. Save checkpoints regularly during training
0. Save the final model as "2048_final_model.pt"
0. Generate a training visualization plot
0. Run an evaluation of the trained agent

### Customizing Training
You can modify the training parameters in the main execution block:

```
trained_agent = train_agent(
    episodes=50000,           # Total training episodes
    visualize_every=1000,     # Visualize gameplay every N episodes
    update_target_every=100   # Update target network frequency
)
```
### Evaluation
To evaluate a trained agent:
```
# Load a previously trained model
agent = DQNAgent(state_shape=(16, 4, 4), action_size=4)
agent.load("2048_best_model.pt")

# Run evaluation with visualization
evaluate_agent(agent, games=5, render=True)
```
