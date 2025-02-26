import pygame
import numpy as np
import random
import time
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
GRID_SIZE = 20
GRID_WIDTH = 20
GRID_HEIGHT = 20
WINDOW_WIDTH = GRID_WIDTH * GRID_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * GRID_SIZE
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Directions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

# Q-Learning parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 1.0
EXPLORATION_DECAY = 0.995
MIN_EXPLORATION_RATE = 0.01

class Snake:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.length = 3
        self.positions = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = random.choice([UP, RIGHT, DOWN, LEFT])
        self.score = 0
        self.steps_without_food = 0
        
    def get_head_position(self):
        return self.positions[0]
    
    def turn(self, direction):
        # Can't turn in the opposite direction
        if self.length > 1 and (direction + 2) % 4 == self.direction:
            return
        self.direction = direction
    
    def move(self):
        head_x, head_y = self.get_head_position()
        
        if self.direction == UP:
            head_y -= 1
        elif self.direction == RIGHT:
            head_x += 1
        elif self.direction == DOWN:
            head_y += 1
        elif self.direction == LEFT:
            head_x -= 1
            
        # Wrap around the edges
        head_x %= GRID_WIDTH
        head_y %= GRID_HEIGHT
        
        self.positions.insert(0, (head_x, head_y))
        
        if len(self.positions) > self.length:
            self.positions.pop()
            
        self.steps_without_food += 1
        
    def check_collision(self):
        head = self.get_head_position()
        return head in self.positions[1:]
    
    def get_food(self, food_position):
        if self.get_head_position() == food_position:
            self.length += 1
            self.score += 1
            self.steps_without_food = 0
            return True
        return False

class SnakeGame:
    def __init__(self, display_ui=True, speed=10):
        self.display_ui = display_ui
        self.speed = speed
        self.snake = Snake()
        self.food_position = self.generate_food()
        self.game_over = False
        
        if display_ui:
            pygame.init()
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption('Snake - AI Learning')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 16)
        
    def generate_food(self):
        position = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
        while position in self.snake.positions:
            position = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
        return position
    
    def get_state(self):
        """
        Creates a state representation for Q-learning.
        State consists of:
        - Danger in each direction (straight, right, left)
        - Direction of movement
        - Food direction (x and y relative positions)
        """
        head_x, head_y = self.snake.get_head_position()
        food_x, food_y = self.food_position
        
        # Direction vectors for checking danger
        if self.snake.direction == UP:
            front_x, front_y = head_x, (head_y - 1) % GRID_HEIGHT
            right_x, right_y = (head_x + 1) % GRID_WIDTH, head_y
            left_x, left_y = (head_x - 1) % GRID_WIDTH, head_y
        elif self.snake.direction == RIGHT:
            front_x, front_y = (head_x + 1) % GRID_WIDTH, head_y
            right_x, right_y = head_x, (head_y + 1) % GRID_HEIGHT
            left_x, left_y = head_x, (head_y - 1) % GRID_HEIGHT
        elif self.snake.direction == DOWN:
            front_x, front_y = head_x, (head_y + 1) % GRID_HEIGHT
            right_x, right_y = (head_x - 1) % GRID_WIDTH, head_y
            left_x, left_y = (head_x + 1) % GRID_WIDTH, head_y
        else:  # LEFT
            front_x, front_y = (head_x - 1) % GRID_WIDTH, head_y
            right_x, right_y = head_x, (head_y - 1) % GRID_HEIGHT
            left_x, left_y = head_x, (head_y + 1) % GRID_HEIGHT
        
        # Check for danger
        danger_straight = (front_x, front_y) in self.snake.positions[1:]
        danger_right = (right_x, right_y) in self.snake.positions[1:]
        danger_left = (left_x, left_y) in self.snake.positions[1:]
        
        # Food direction relative to head
        food_up = food_y < head_y
        food_right = food_x > head_x
        food_down = food_y > head_y
        food_left = food_x < head_x
        
        # Direction of movement
        dir_up = self.snake.direction == UP
        dir_right = self.snake.direction == RIGHT
        dir_down = self.snake.direction == DOWN
        dir_left = self.snake.direction == LEFT
        
        # Create a binary state representation
        state = (
            danger_straight, danger_right, danger_left,
            dir_up, dir_right, dir_down, dir_left,
            food_up, food_right, food_down, food_left
        )
        
        return state
    
    def get_reward(self, prev_state, action, new_state):
        """Calculate reward for the agent's action"""
        # Check if we've hit ourselves
        if self.snake.check_collision():
            return -10  # Big penalty for dying
        
        # Check if we got food
        if self.snake.get_head_position() == self.food_position:
            return 10  # Big reward for getting food
        
        # Small penalty for each step to encourage efficiency
        if self.snake.steps_without_food > 100:
            return -1  # Stronger penalty if we're not finding food
            
        # Calculate reward based on whether we're moving toward or away from food
        head_x, head_y = self.snake.get_head_position()
        food_x, food_y = self.food_position
        
        old_distance = abs(self.old_head_pos[0] - food_x) + abs(self.old_head_pos[1] - food_y)
        new_distance = abs(head_x - food_x) + abs(head_y - food_y)
        
        if new_distance < old_distance:
            return 0.1  # Small reward for moving toward food
        else:
            return -0.1  # Small penalty for moving away from food
    
    def step(self, action):
        # Remember old position for reward calculation
        self.old_head_pos = self.snake.get_head_position()
        
        # Map action (0, 1, 2) to direction changes
        # 0 = straight, 1 = right turn, 2 = left turn
        if action == 1:  # Turn right
            self.snake.turn((self.snake.direction + 1) % 4)
        elif action == 2:  # Turn left
            self.snake.turn((self.snake.direction - 1) % 4)
        # action 0 = continue straight (no turn)
        
        # Move the snake
        self.snake.move()
        
        # Check if got food
        if self.snake.get_food(self.food_position):
            self.food_position = self.generate_food()
        
        # Check if game is over (collision)
        if self.snake.check_collision() or self.snake.steps_without_food > 200:
            self.game_over = True
        
        return self.get_state(), self.get_reward(None, action, None), self.game_over
    
    def reset(self):
        self.snake.reset()
        self.food_position = self.generate_food()
        self.game_over = False
        self.old_head_pos = self.snake.get_head_position()
        return self.get_state()
    
    def render(self):
        if not self.display_ui:
            return
            
        self.screen.fill(BLACK)
        
        # Draw snake
        for position in self.snake.positions:
            x, y = position
            pygame.draw.rect(self.screen, GREEN, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))
            
        # Draw food
        food_x, food_y = self.food_position
        pygame.draw.rect(self.screen, RED, (food_x * GRID_SIZE, food_y * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        
        # Draw score
        score_text = self.font.render(f'Score: {self.snake.score}', True, WHITE)
        self.screen.blit(score_text, (5, 5))
        
        pygame.display.update()
        self.clock.tick(self.speed)
        
    def handle_events(self):
        if not self.display_ui:
            return True
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

class QLearningAgent:
    def __init__(self):
        self.q_table = {}  # Dictionary to store Q-values for state-action pairs
        self.learning_rate = LEARNING_RATE
        self.discount_factor = DISCOUNT_FACTOR
        self.exploration_rate = EXPLORATION_RATE
        self.exploration_decay = EXPLORATION_DECAY
        self.min_exploration_rate = MIN_EXPLORATION_RATE
        
    def get_q_value(self, state, action):
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0.0
        return self.q_table[(state, action)]
    
    def choose_action(self, state):
        if random.random() < self.exploration_rate:
            return random.randint(0, 2)  # Explore: 0=straight, 1=right, 2=left
        else:
            # Exploit: choose action with highest Q-value
            q_values = [self.get_q_value(state, a) for a in range(3)]
            return np.argmax(q_values)
    
    def update(self, state, action, reward, next_state):
        # Q-learning update equation
        old_q = self.get_q_value(state, action)
        
        # Get maximum Q-value for next state
        next_q_values = [self.get_q_value(next_state, a) for a in range(3)]
        max_next_q = max(next_q_values)
        
        # Q-learning formula
        new_q = old_q + self.learning_rate * (reward + self.discount_factor * max_next_q - old_q)
        
        # Update Q-table
        self.q_table[(state, action)] = new_q
        
    def decay_exploration(self):
        # Reduce exploration rate over time
        self.exploration_rate = max(self.min_exploration_rate, 
                                    self.exploration_rate * self.exploration_decay)

def train_agent(episodes=1000, display_ui=False):
    env = SnakeGame(display_ui=display_ui, speed=30 if display_ui else 0)
    agent = QLearningAgent()
    
    scores = []
    avg_scores = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            # Choose action
            action = agent.choose_action(state)
            
            # Take action and observe next state and reward
            next_state, reward, done = env.step(action)
            
            # Update Q-table
            agent.update(state, action, reward, next_state)
            
            total_reward += reward
            state = next_state
            
            if display_ui:
                env.render()
                if not env.handle_events():
                    pygame.quit()
                    return
            
            if done:
                break
        
        # Decay exploration rate
        agent.decay_exploration()
        
        # Track score
        scores.append(env.snake.score)
        avg_scores.append(np.mean(scores[-100:]) if len(scores) > 100 else np.mean(scores))
        
        if episode % 100 == 0:
            print(f"Episode: {episode}, Score: {env.snake.score}, Avg Score: {avg_scores[-1]:.2f}, " +
                  f"Exploration: {agent.exploration_rate:.2f}")
    
    if display_ui:
        pygame.quit()
    
    return agent, scores, avg_scores

def plot_training_results(scores, avg_scores):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    plt.subplot(1, 2, 2)
    plt.plot(avg_scores)
    plt.title('Average Scores (last 100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

def demonstrate_agent(agent, episodes=5):
    env = SnakeGame(display_ui=True, speed=10)
    
    for episode in range(episodes):
        state = env.reset()
        
        while True:
            # Choose action (no exploration)
            q_values = [agent.get_q_value(state, a) for a in range(3)]
            action = np.argmax(q_values)
            
            # Take action
            next_state, _, done = env.step(action)
            state = next_state
            
            env.render()
            if not env.handle_events():
                pygame.quit()
                return
            
            if done:
                print(f"Episode {episode+1} Score: {env.snake.score}")
                time.sleep(1)
                break
    
    pygame.quit()

if __name__ == "__main__":
    # Train the agent
    print("Training the agent...")
    agent, scores, avg_scores = train_agent(episodes=1000, display_ui=False)
    
    # Plot training results
    plot_training_results(scores, avg_scores)
    
    # Demonstrate trained agent
    print("\nDemonstrating trained agent...")
    demonstrate_agent(agent)