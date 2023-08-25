import torch
import random
import numpy as np
from collections import deque
from snake_game import SnakeGame, Direction, Point, BLOCK_SIZE
from model import Linear_QN, Trainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000


class Worker:
    game = SnakeGame()

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QN(11, 121, 121, 3)
        self.trainer = Trainer(self.model)


    def get_state(self, game):
        head = game.snake[0]
        xpt_l = Point(head.x - BLOCK_SIZE, head.y)
        xpt_r = Point(head.x + BLOCK_SIZE, head.y)
        xpt_u = Point(head.x, head.y - BLOCK_SIZE)
        xpt_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [

            (dir_r and game.is_collision(xpt_r)) or
            (dir_l and game.is_collision(xpt_l)) or
            (dir_u and game.is_collision(xpt_u)) or
            (dir_d and game.is_collision(xpt_d)),

            (dir_r and game.is_collision(xpt_d)) or
            (dir_l and game.is_collision(xpt_u)) or
            (dir_u and game.is_collision(xpt_r)) or
            (dir_d and game.is_collision(xpt_l)), 

            (dir_l and game.is_collision(xpt_d)) or 
            (dir_r and game.is_collision(xpt_u)) or
            (dir_u and game.is_collision(xpt_l)) or 
            (dir_d and game.is_collision(xpt_r)),

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]

        return np.array(state, dtype=int)

    def save_state(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over)) 

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)
    

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


