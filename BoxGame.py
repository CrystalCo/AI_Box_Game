# Crystal Contreras     SPRING 2020 CSC-480
import random
import math
import heapq

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
        self.random_corner_edge_dict = [[len(self.state) - 1, 'right_edge'], [len(self.state) - 1, 'bottom_edge'], [0, 'top_edge'], [0, 'left_edge']]
    
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
                'edges':    {
                    'top_edge':    0,
                    'bottom_edge': 0,
                    'right_edge':  0,
                    'left_edge':   0
                },
                'box_closed':   False
            }
            print("Box {}: {}".format(i, box))
            initial_state.append(box)

        return initial_state

    def _get_random_num(self):
        return random.randint(1, 5)

    def _get_human_move(self, state):
        legal_moves = self._get_possible_moves(state)
        print("\nLegal Moves:\n{}".format(legal_moves))
        
        # Read human's move input
        value_in = input("Enter the box id and edge from the legal moves returned. Separate answer with a space. " + 
        "Example:   2 top_edge " +
        "Edges selected that are filled (1) will forfeit your turn. \n")
        inputs = value_in.split()
        box = int(inputs[0])
        edge = inputs[1]

        print("\nYour input was: {} {} ".format(box, edge))

        # If it is legal, make that move (change assigned edge state)
        self._set_edge(box, edge)
        
    def _get_AI_move(self, state):
        """ AKA Minimax.  Computates a move, then makes the move it sees best. """
        move = self.MiniMax_Decision(state)
        print("AI move: ", move)
        self._set_edge(move[0], move[1])
        return

    def MiniMax_Decision(self, state):
        """ Returns an action """
        # Compute the element a of set S that has a maximum value of f(a)
        legal_moves = self._get_possible_moves(state)
        print("\nLegal Moves:\n{}".format(legal_moves))

        # Rank boxes that only have 1 edge left as top priorities.
        # This gives us a chance to reduce the amount of checks we do below.
        # If I move this below will it automatically break after the first instance of this?
        for m in legal_moves:
            if len(m['edges']) == 1:
                action = [m['box_id'], m['edges'].pop()]
                return action
        
        # TODO: Choose edge at random if all util vals are 0 OR 
        # If most of the boxes are empty, return a random edge to save time
        if self._get_random_edge(state):
            return self.random_corner_edge_dict.pop()

        # default returns min at front. To get max, mult val by -
        pq = []
        # Tie-breaker counter. 
        pq_c = 0

        # Sort the boxes by weight
        weighted_legal_moves = []
        for m in legal_moves:
            heapq.heappush(weighted_legal_moves, (m['weight'] * -1, m['box_id'], m))

        # To reduce the amount of checks we do for each move, 
        # let us pop boxes from our PQ n times, where n <= ply limit 
        temp_ply_count = self.ply_limit

        while temp_ply_count > 0:
            current_box = heapq.heappop(weighted_legal_moves)
            current_box = current_box[2]
            boxID = current_box['box_id']
    
            # For each action available, return the utility value 
            for m in current_box['edges']:
                if m == 'top_edge':
                    h = self.ply_limit
                    action = [boxID, m]
                    t_val = self._min_value(self._value(state, action, h))
                    heapq.heappush(pq, (t_val * -1, pq_c, action))
                    pq_c += 1
                if m == 'bottom_edge':
                    h = self.ply_limit
                    action = [boxID, m]
                    b_val = self._min_value(self._value(state, action, h))
                    heapq.heappush(pq, (b_val * -1, pq_c, action))                     
                    pq_c += 1
                if m == 'right_edge':
                    h = self.ply_limit
                    action = [boxID, m]
                    r_val = self._min_value(self._value(state, action, h))
                    heapq.heappush(pq, (r_val * -1, pq_c, action))                     
                    pq_c += 1
                if m == 'left_edge':
                    h = self.ply_limit
                    action = [boxID, m]
                    l_val = self._min_value(self._value(state, action, h))
                    heapq.heappush(pq, (l_val * -1, pq_c, action))                     
                    pq_c += 1

        # Return max
        action = heapq.heappop(pq) 
        action = action[2]
        return action

    def _min_value(self, state, horizon):
        """ returns a utility value """
        # v = infinity
        v = 999999
        legal_moves = self._get_possible_moves(state)
        for move in legal_moves:
            v = max(v, self._min_value(self._value(state, move, horizon - 1)))
        return v 

    def _get_random_edge(self, state):
        """ Returns a random corner edge for the AI to choose from to save time 
        on checking in the beginning of the game when lots of moves are available. """
        half_check = int(len(state)/2)
        half_check_count = 0
        for m in range(half_check):
            if len(state[m]['edges']) >= 3:
                half_check_count += 1

        if half_check == half_check_count:
            return True
        else:
            return False

    def _get_possible_moves(self, state):
        '''  Returns a list of available edges. '''
        available_edges = []
        # Need a counter for index of available edges since boxes that are closed will be skipped
        available_boxes_counter = 0
        for i in state:
            if i['box_closed'] == False:
                available_edges.append({
                    'box_id': i['box_id'],
                    'weight': i['weight'],
                    'edges': [] 
                })

                for e, val in i['edges'].items():
                    if val == 0:
                        available_edges[available_boxes_counter]['edges'].append(e)

                available_boxes_counter += 1

        return available_edges

    def _get_edge_val(self, box, edge):
        """ Returns the value within the box's edge selected. """
        edge_value = self.state[box]['edges'][edge]
        return edge_value

    def _box_filled(self, box):
        """ 'Closes' box by changing box_close to True if box is filled. """
        if (self.state[box]['edges']['top_edge'] == 1) & (self.state[box]['edges']['bottom_edge'] == 1) & (self.state[box]['edges']['left_edge'] == 1) & (self.state[box]['edges']['right_edge'] == 1):
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
            self.state[box]['edges'][edge] = 1
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
        # Corner cases
        if box == 0:
            if edge == 'right_edge':
                self._set_edge(box+1, 'left_edge')
            elif edge == 'bottom_edge':
                x = int(math.sqrt(boxes))
                self._set_edge(x, 'top_edge')
        elif box == last_box_index:
            if edge == 'left_edge':
                self._set_edge(last_box_index - 1, 'right_edge')
            elif edge == 'top_edge':
                box_above_index = int(boxes - math.sqrt(boxes) - 1)
                self._set_edge(box_above_index, 'bottom_edge')
        # Middle Cases
        elif self._is_double_edge(box, edge):
            new_box, new_edge = self.adj_edge_dict[edge]
            new_box += box
            self._set_edge(new_box, new_edge)

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

    def Play_game(self):
        """ Plays the game until there is a winner or human terminates game. """
        state = self.state

        while(self.game_on):
            if self._player():
                # Human's turn
                self._get_human_move(state)
            else:
                # AI's turn
                self._get_AI_move(state)
                
            self._terminal_test(self.state)
            if(self.game_on):
                self._player_switch()

    def _make_node(self, state, horizon):
        ''' Creates 1st ply node '''
        node = {
            'state': state,
            'path_cost': 0,
            'ply_limit': horizon
        }
        return node    

    def _player(self):
        """ 
        Defines which player has the move in a state. 
        True  = Human;      False = AI
        Player max = human;   Player min = AI 
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
        """Returns the set of legal moves in a state.
        Same as _get_possible_moves()"""
    
    def _result(self, state, action):
        """The transition model, which defines the result of a move
        OR returns a number representing score after move."""
        # Do we need 'state' if _set_edge already changes self.state?
        # action = (box, edge)
        self._set_edge(action[0], action[1])
        # Does it make a diff' if we do 2 if statements vs if/else?
        if self._player():
            return self.player_max_score
        else:
            return self.player_min_score
    
    def _value(self, state, action, horizon):
        """ same as _result() OR Returns util val
        """
        if self._cutoff_test(state, horizon):
            return self._evaluation(state)

        # action = (box, edge)
        self._set_edge(action[0], action[1])
        # Does it make a diff' if we do 2 if statements vs if/else?
        # Does it make a diff' if we start 2 with "next player" vs "current player"?
        if self._player():
            return self._max_value()
        else:
            return self._min_value()


    def _terminal_test(self, state):
        """ A terminal test, which is true when the game is over and false otherwise. 
        States where the game has ended are called terminal states.
        Terminal state is when all edges are filled/popped.

        Replace the terminal test by a cutoff test that decides when to apply EVAL.
        """
        if self._get_possible_moves(state) == []:
            if self.player_max_score > self.player_min_score:
                print("Player Max Won!")
            elif self.player_max_score < self.player_min_score:
                print("Player Min Won!")
            else:
                print("Tie!")
            self.game_on = False
            return True
        else:
            return False
        
    def _cutoff_test(self, state, horizon):
        """ Uses ply limit (aka horizon) to determine when to stop and evaluate the utility up to that point. """
        if horizon <= 0:
            return True
        else:
            return False

    def _evaluation(self, state):
        """
        A modified utility function.
        A payoff function. Defines the final numeric value for a game
        using heuristic evaluation function A*, which estimates the position’s utility
        """
        if self._player():
            return self.player_max_score
        else:
            return self.player_min_score





"""
the suggestion is to alter minimax in two ways: 

1. Replace the utility function by a heuristic evaluation function EVAL, which estimates the position’s utility; and
2. Replace the terminal test by a cutoff test that decides when to apply EVAL.

"""
        
easy_2x2 = 4   # boxes

game = BoxGame(easy_2x2, 2)
game.Play_game()

