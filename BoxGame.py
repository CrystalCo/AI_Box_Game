# Crystal Contreras     SPRING 2020 CSC-480
import random
import math

# in check functions, keep in mind alpha-beta pruning 
# to set up 

class BoxGame:
    # When the game starts the human player will specify the board size and ply limit
    def __init__(self, state, ply_limit, player=1):
        """ The size of the board is the number of boxes to fill
        The state of the board with randomly generated weights """
        self.state = self._generate_board(state)
        # how many plys the AI will search (i.e., the horizon for the minimax)
        self.ply_limit = ply_limit
        # Player max = human
        self.player_max_score = 0
        # Player min = AI 
        self.player_min_score = 0
        # If player = 1, it is human's turn. Player = 0 == AI turn
        self.player = player
        self.game_on = True
        self.adj_edge_dict = {
            'top_edge': (int(-len(self.state)**0.5), 'bottom_edge'),
            'bottom_edge': (int(len(self.state)**0.5), 'top_edge'),
            'left_edge': (-1, 'right_edge'),
            'right_edge': (1, 'left_edge')
        }
    
    def _generate_board(self, boxes):
        """ 
        Initializes a board with objects representing boxes.
        Each box contains a weight and 4 edges.
        """
        initial_state = []  # array of boxes

        for i in range(boxes):
            box = {
                'box_id':      i,
                'weight':      self._get_random_num(),
                'top_edge':    0,
                'bottom_edge': 0,
                'right_edge':  0,
                'left_edge':   0,
                'box_closed':   False
            }
            print("Box {}: {}".format(i, box))
            initial_state.append(box)

        return initial_state

    def _get_random_num(self):
        return random.randint(1, 5)

    def _get_human_move(self):
        # TODO: Return list of possible moves instead of board state
        print("\nBoard state:")
        # print("\nBoard state:\n{}\n".format(self.state))
        legal_moves = self._get_possible_moves(self.state)
        print(legal_moves)
        
        # Read human's move input
        value_in = input("Enter the box id and edge you want to flip. Separate answer with a space. " + 
        "For example, to flip box id 2's top edge, enter '2 top_edge'. " +
        "An edge with state 0 means that edge is available to flip. " +
        "Edges selected that are filled (1) will return an error. \n")
        inputs = value_in.split()
        box = int(inputs[0])
        edge = inputs[1]

        print("\nYour input was: {} {} ".format(box, edge))

        # If it is legal, make that move (change assigned edge state)
        self._set_edge(box, edge)

    def _get_edge_val(self, box, edge):
        """ Returns the value within the box's edge selected. """
        edge_value = self.state[box][edge]
        return edge_value

    def _box_filled(self, box):
        """ 'Closes' box by changing box_close to True if box is filled. """
        if (self.state[box]['top_edge'] == 1) & (self.state[box]['bottom_edge'] == 1) & (self.state[box]['left_edge'] == 1) & (self.state[box]['right_edge'] == 1):
            self.state[box]['box_closed'] = True
            self._add_points(box)
        else:
            return

    def _add_points(self, box):
        """ Adds points to player """
        if self._player():
            self.player_max_score += self.state[box]['weight']
            print("Human scores.  Human total: {}. AI total: {}.\n".format(self.player_max_score, self.player_min_score))
        else:
            self.player_min_score += self.state[box]['weight']
            print("AI scores.  Human total: {}. AI total: {}.\n".format(self.player_max_score, self.player_min_score))

    def _set_edge(self, box, edge):
        """ Flip edge state from 0 (available) to 1 (filled) """
        if not self._get_edge_val(box, edge):
            self.state[box][edge] = 1
            # Check if box filled
            self._box_filled(box)
            print("New state of box: {}.\n".format(self.state[box]))
            # Check adjacent edges
            self._set_adj_edges(box, edge)
        else:
            # Prints illegal move if edge is already filled.
            # print("Illegal move")
            # This is also a way for _set_adj_edges & _set_edge to break out of an infinite loop 
            return

    def _set_adj_edges(self, box, edge):
        """ Marks off adjacent edges """
        boxes = len(self.state)
        if boxes == 1:
            # single box board doesn't have adjacent boxes to check 
            return
        last_box_index = boxes - 1
        # Edge cases
        if box == 0:
            if edge == 'right_edge':
                self._set_edge(box+1, 'left_edge')
                # self.state[box+1]['left_edge'] = 1
            elif edge == 'bottom_edge':
                x = int(math.sqrt(boxes))
                # self.state[x]['top_edge'] = 1
                self._set_edge(x, 'top_edge')
        elif box == last_box_index:
            if edge == 'left_edge':
                # self.state[last_box_index - 1]['right_edge'] = 1
                self._set_edge(last_box_index - 1, 'right_edge')
            elif edge == 'top_edge':
                box_above_index = int(boxes - math.sqrt(boxes) - 1)
                # self.state[box_above_index]['bottom_edge'] = 1
                self._set_edge(box_above_index, 'bottom_edge')
        # Middle Cases
        elif self._is_double_edge(box, edge):
            new_box, new_edge = self.adj_edge_dict[edge]
            new_box += box
            self._set_edge(new_box, new_edge)
        # print("Adjacent edges filled in: {}\n".format(self.state))

    def _is_double_edge(self, box, edge):
        """ 
        Checks adjacent edges in the middle of the board 
        Returns True if the edges touch an adjacent box; returns False otherwise.
        """
        result = True
        # Another way to get sqrt of total boxes on board
        s = int(len(self.state) ** 0.5)
        # 1st row
        if box < s and edge == 'top_edge':
            result = False
        # Last row
        if box >= s * (s-1) and edge == 'bottom_edge':
            result = False
        # Left row
        if box % s == 0 and edge == 'left_edge':
            result = False
        # Right row
        if (box + 1) % s == 0 and edge == 'right_edge':
            result = False
        return result

    def _get_possible_moves(self, state):
        '''  Returns a list of available edges. '''
        available_edges = []
        # Need a counter for index of available edges since boxes that are closed will be skipped
        available_boxes_counter = 0
        for i in state:
            if i['box_closed'] == False:
                available_edges.append({
                    'box_id': i['box_id'],
                    'weight': i['weight']
                })
                if i['top_edge'] == 0:
                    available_edges[available_boxes_counter]['top_edge'] = 0
                if i['bottom_edge'] == 0:
                    available_edges[available_boxes_counter]['bottom_edge'] = 0
                if i['right_edge'] == 0:
                    available_edges[available_boxes_counter]['right_edge'] = 0
                if i['left_edge'] == 0:
                    available_edges[available_boxes_counter]['left_edge'] = 0
                available_boxes_counter += 1
        return available_edges

    def Play_game(self):
        """ Plays the game until there is a winner or human terminates game. """

        while(self.game_on):
            if self._player:
                self._get_human_move()
            else:
                self._get_human_move()
                
            self._terminal_test(self.state)
            self._player_switch()
        
    def _player(self):
        """ 
        Defines which player has the move in a state. 
        True  = Human
        False = AI 
        """
        return self.player

    def _player_switch(self):
        """ 
        Defines which player has the move in a state. 
        True  = Human
        False = AI 
        """
        if self.player:
            self.player = 0
            print("AI's turn now.\n")
        else:
            self.player = 1
            print("Human's turn now.\n")

    def _actions(self, state):
        """Returns the set of legal moves in a state."""
    
    def _result(self, state, action):
        """The transition model, which defines the result of a move"""

    def _terminal_test(self, state):
        """ A terminal test, which is true when the game is over and false otherwise. 
        States where the game has ended are called terminal states.
        Terminal state is when all edges are filled/popped.
        """
        # if (self._get_possible_moves(state) = 0):
        #     if self.player_max_score > self.player_min_score:
        #         print("Player Max Won!")
        #     elif self.player_max_score < self.player_min_score:
        #         print("Player Min Won!")
        #     else:
        #         print("Tie!")
        #     self.game_on = False
        #     return True
        # else:
        return False


    def _utility(self, state, player):
        """A payoff function.  Defines the final numeric value for a game 
        that ends in terminal state s for a player p. 
        Outcome is win, loss, or draw (+1, 0, 1/2)
        """

        
easy_2x2 = 4   # boxes

game = BoxGame(easy_2x2, 2)
game.Play_game()

