from snake_game import SnakeGame
from graph_plot import plot
from worker import Worker



def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    worker = Worker()
    game = SnakeGame()
    while True:
        curr_state = worker.get_state(game)

        final_move = worker.get_action(curr_state)

        reward, game_over, score = game.play_step(final_move)
        state_new = worker.get_state(game)

        worker.train_short_memory(curr_state, final_move, reward, state_new, game_over)

        worker.save_state(curr_state, final_move, reward, state_new, game_over)

        if game_over:
    
            game.reset()
            worker.n_games += 1
            worker.train_long_memory()

            if score > record:
                record = score
                worker.model.save()

            print('Game', worker.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / worker.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()