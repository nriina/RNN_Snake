import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.twodim_base import eye

class snake_game():

    def __init__(self,window):
        self.x_len = window[1]
        self.y_len = window[0]
        self.initial_snake_location = [3,3]
        self.snake = [self.initial_snake_location]
        self.food_location = [3,5]
        self.snake_length = 1
        self.snake_dir = 'right' #'right', '
        self.dir_options = [
            'left',
            'right',
            'up',
            'down'
        ]
    
        self.game_graph = np.zeros((self.x_len,self.y_len))
        self.game_over = False
        self.turn_length = 0
        self.return_stats = [0,0]
        self.old_graph = np.zeros((self.x_len,self.y_len))

    def reset_game_graph(self):
        self.game_graph = np.zeros((self.x_len,self.y_len))

    def place_food(self):
        old_spot = self.food_location
        new_spot_list = np.random.randint(len(self.game_graph),size=2)
        new_spot = [new_spot_list[0],new_spot_list[1]]
        assign = True
        while assign:
            if new_spot != old_spot:
                if not new_spot in self.snake:
                    self.food_location = new_spot
                    assign = False
    
    def check_board(self):

        if self.snake[0][0] < 0:
            self.game_over = True
        elif self.snake[0][0] > self.x_len-1:
            self.game_over = True
        elif self.snake[0][1] < 0:
            self.game_over = True
        elif self.snake[0][1] > self.y_len-1:
            self.game_over = True  
        else:
            # copy_snake = self.snake.copy()
            # print('whole snak',self.snake)
            # print('self snake',self.snake[0][0])
            # print('snake after',self.snake[0][1:])
            if len(self.snake) > 1:
                if self.snake[0] in self.snake[1:]:
                    # print(' in snake')
                    self.game_over=True
        # elif self.snake[0]

            # self.turn_length +=1
        
        if self.game_over == True:
            self.end_game()
    
    def update_board(self):
        if self.snake[0] == self.food_location:
            # print('same loc')
            self.snake_length +=1
            self.place_food()
        self.check_board()
        if len(self.snake) > 1:
            if self.snake[1] == self.food_location:
                print('its in your neck you rat bastard')

    def move(self):
        self.update_board()
        if self.snake_dir == 'right':
            old_point = self.snake[0]
            self.snake.insert(0,[old_point[0] + 1, old_point[1]])
            if len(self.snake) > self.snake_length:
                self.snake.pop()
        elif self.snake_dir == 'left':
            old_point = self.snake[0]
            self.snake.insert(0,[old_point[0] - 1, old_point[1]])
            if len(self.snake) > self.snake_length:
                self.snake.pop()  
        elif self.snake_dir == 'up':
            old_point = self.snake[0]
            self.snake.insert(0,[old_point[0], old_point[1] + 1])
            if len(self.snake) > self.snake_length:
                self.snake.pop()  
        elif self.snake_dir == 'down':
            old_point = self.snake[0]
            self.snake.insert(0,[old_point[0], old_point[1] - 1])
            if len(self.snake) > self.snake_length:
                self.snake.pop() 
                
        self.update_board()

        # print('total loc',self.snake[0])
        # print('current loc',self.snake[0][0])
        # print('current loc2',self.snake[0][1])
        self.check_board()
        self.turn_length +=1
        

    def end_game(self):
        self.return_stats = [self.snake_length, self.turn_length]
    
    def show_game(self,print_Graph=True):
        self.update_board()
        self.check_board()
        if not self.game_over:
            # print('post game over status:',self.game_over)
            wall = self.x_len - 1
            rep_snake_loc = ((wall-self.snake[0][1]), self.snake[0][0])
            rep_food_loc = ((wall-self.food_location[1]), self.food_location[0])
            # print('rep snake',rep_snake_loc)
            # print('rep food',rep_food_loc)
            if rep_snake_loc == rep_food_loc:
                # print('same loc')
                self.snake_length +=1
                self.place_food()
            # print('snake loc',rep_snake_loc)
            # self.game_graph[rep_snake_loc] = 1
            for i in range(0,len(self.snake)): #need to make snake history for the tail
                rep_snake_loc = ((wall-self.snake[i][1]), self.snake[i][0])
                self.game_graph[rep_snake_loc] = 0.8

            self.game_graph[rep_food_loc] = -0.5

        if self.game_over == False:
            if print_Graph == True:
                fig, axs = plt.subplots()
                # vegetables = decay_iterations #for when we want labels
                # farmers = thresh_iterations
                vegetables = [] #empty lists for no labels
                farmers = [] #empty lists for no labels
                # print('snake loc in show ',self.snake[0])
                # print('self x len',self.x_len)
                # print('game over status:',self.game_over)
                
                axs.imshow(self.game_graph,cmap ='bwr')
                axs.set_title('Snake.Riina, Red=Snake, Blue=Food')

                # We want to show all ticks:
                axs.set_xticks(np.arange(len(farmers)))
                axs.set_yticks(np.arange(len(vegetables)))
                plt.show()
                self.old_graph = self.game_graph
                self.reset_game_graph()
            else:
                self.end_game()
        # else:
            # print('game is over doofensmerf')

    def key_to_turn(self):
        prev_move = self.snake_dir
        good_move = False
        move = input(' u = up, h= left, j= down, k=right')
        if prev_move == 'up':
            if move == 'j':
                print('invalid move, try again')
            else:
                good_move = True
        elif prev_move == 'down':
            if move == 'u':
                print('invalid move, try again')
            else:
                good_move = True
        elif prev_move == 'left':
            if move == 'k':
                print('invalid move, try again')
            else:
                good_move = True
        elif prev_move == 'right':
            if move == 'h':
                print('invalid move, try again')
            else:
                good_move = True
        ###
        if good_move == True:
            if move == 'u':
                self.snake_dir = 'up'
            elif move == 'h':
                self.snake_dir = 'left'
            elif move == 'j' :
                self.snake_dir = 'down'
            elif move == 'k':
                self.snake_dir = 'right'

    def val_to_turn(self,n, list_vals): #n is how many inputs, list vals is the list of values, n can be =1 or =4

        if int(n) == 1:
            output = float(list_vals[0])
            if output < 0.25:
                new_dir = 'up'
            if output >= 0.25:
                if output < 0.5:
                    new_dir = 'left'
            if output >= 0.5:
                if output < 0.75:
                    new_dir = 'right'
            if output >= 0.75:
                new_dir = 'down'
            prev_move = self.snake_dir
            good_move = False
            # print('new dir',new_dir)
            if prev_move == 'up':
                if new_dir != 'down':
                    good_move = True
            elif prev_move == 'down':
                if new_dir != 'up':
                    good_move = True
            elif prev_move == 'left':
                if new_dir != 'right':
                    good_move = True
            elif prev_move == 'right':
                if new_dir != 'left':
                    good_move = True
                
            if good_move == True:
                self.snake_dir = new_dir
            
        elif int(n) == 4:
            max_val = max(list_vals)
            val_index = list_vals.index(max_val)
            if val_index == 0:
                new_dir = 'up'
            elif val_index == 1:
                new_dir = 'left'
            elif val_index == 2:
                new_dir = 'right'
            elif val_index == 3:
                new_dir = 'down'
            prev_move = self.snake_dir
            good_move = False
            if prev_move == 'up':
                if new_dir != 'down':
                    good_move = True
            elif prev_move == 'down':
                if new_dir != 'up':
                    good_move = True
            elif prev_move == 'left':
                if new_dir != 'right':
                    good_move = True
            elif prev_move == 'right':
                if new_dir != 'left':
                    good_move = True
            if good_move == True:
                self.snake_dir = new_dir

        else:
            print('invalid n, must be 1 or 4')
        
        

    def see_board(self,view='all'): #view can be 'all','local'
        # self.show_game(print_Graph=False)
        # print('old graph',self.old_graph)
        # print('game graph',self.game_graph)
        wall = self.x_len - 1
        if view == 'all':
            flat_graph = []
            for row in self.game_graph:
                for spot in row:
                    flat_graph.append(spot)
            # print('self.game_graph',len(flat_graph))
            return flat_graph
        elif view == 'local':
            head = self.snake[0]
            food = self.food_location
            eye_width = 3
            sight = list(np.ones((eye_width,eye_width)))
            rep_snake_loc = ((wall-self.snake[0][1]), self.snake[0][0])
            rep_food_loc = ((wall-self.food_location[1]), self.food_location[0]) #need to make this into function
            ## update right infront
            print('snake loc',rep_snake_loc)
            print('rep food',rep_food_loc)
            # print('game graph snake',self.game_graph[rep_snake_loc])
            # print('old graph snake',self.old_graph[rep_snake_loc])
            sight[0][int(eye_width/2)] = self.game_graph[rep_snake_loc]
            sight[1][int(eye_width/2)] = self.game_graph[rep_snake_loc[0] -1,rep_snake_loc[1]] #at least this is here
            sight[2][int(eye_width/2)] = self.game_graph[rep_snake_loc[0] -2,rep_snake_loc[1]] #eyes need to turn when head turns

            sight[1][int(eye_width/2)+1] = self.game_graph[rep_snake_loc[0]-1,rep_snake_loc[1] + 1]
            sight[2][int(eye_width/2)+1] = self.game_graph[rep_snake_loc[0]-2,rep_snake_loc[1] + 1]

            sight[1][int(eye_width/2)-1] = self.game_graph[rep_snake_loc[0]-1,rep_snake_loc[1] - 1]
            sight[2][int(eye_width/2)-1] = self.game_graph[rep_snake_loc[0]-2,rep_snake_loc[1] - 1]
            for row in range(len(sight)):
                sight[row] = list(sight[row])
            print('sight',sight)
            # flat_graph = []
            # for row in self.old_graph:
            #     for spot in row:
            #         flat_graph.append(spot)

        else:
            print('view must be equal to "all" or "local"')



if __name__ == "__main__":
    game1 = snake_game(window=(10,10))
    print('initial snake',game1.snake[0])
    print('initial food',game1.food_location)
    nn_Val = [0.6,0.2,0.5,0.4]
    # game1.show_game()
    while not game1.game_over:
        # print('snake dir',game1.snake_dir)
        # next_dir = 
        game1.key_to_turn()
        # game1.val_to_turn(4,nn_Val)
        # nn_Val.insert(0,nn_Val[0]+0.2)
        # nn_Val.pop()
        game1.move()
        # next_dir = np.random.choice(game1.dir_options)
        # game1.snake_dir = next_dir
        viewboard = game1.see_board(view='all')
        print('view',viewboard)
        
        if not game1.game_over:
            print('snek loc',game1.snake)
            print('food loc',game1.food_location)
            game1.show_game()

    print('final length',game1.return_stats[0])
    print('turn length',game1.return_stats[1])
        # game1.move()
    # # game1.place_food()
    # game1.show_game()
    # game1.snake_dir = 'up'
    # # game1.show_game()
    # game1.move()
    # # game1.show_game()
    # game1.move()
    # game1.show_game()
    # # game1.snake_dir = 'up'
    # game1.move()
    # game1.show_game()
    # game1.snake_dir = 'left'
    # game1.move()
    # game1.show_game()
    # game1.move()   
    # # game1.snake_dir = 'up'
    # game1.move()
    # game1.show_game()
    # game1.move()
    # game1.show_game()
    
    