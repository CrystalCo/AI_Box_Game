# Crystal Contreras     SPRING 2020 CSC-480
import random
import math
import heapq
import numpy as np
import copy

class BoxGame:
    # When the game starts the human player will specify the board size and ply limit
    def __init__(self, state, ply_limit, player=1):
        """ 
            The size of the board is the number of boxes to fill
            The state of the board with randomly generated weights 
        """
        self.state = self._generate_board(state)
        self.box_length = math.sqrt(state)
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
  
    def _add_points(self, box):
        """ Adds points to player """
        if self._player():
            self.player_max_score += self.state[box]['weight']
            print("Human scores.  Human total: {}. AI total: {}.".format(self.player_max_score, self.player_min_score))
        else:
            self.player_min_score += self.state[box]['weight']
            print("AI scores.  Human total: {}. AI total: {}.\n".format(self.player_max_score, self.player_min_score))

    def _add_points_copy(self, node, box):
        """ Adds points to player """
        if node['player']:
            node['player_max_score'] += node['state'][box]['weight']
            print("Human scores.  Human total: {}. AI total: {}.\n".format(node['player_max_score'], node['player_min_score']))
            return node
        else:
            node['player_min_score'] += node['state'][box]['weight']
            print("AI scores.  Human total: {}. AI total: {}.\n".format(node['player_max_score'], node['player_min_score']))
            return node

    def _box_filled(self, box):
        """ 'Closes' box by changing box_close to True if box is filled. """
        if (self.state[box]['edges']['top_edge'] == 1) & (self.state[box]['edges']['bottom_edge'] == 1) & (self.state[box]['edges']['left_edge'] == 1) & (self.state[box]['edges']['right_edge'] == 1):
            self.state[box]['box_closed'] = True
            self._add_points(box)
        else:
            return

    def _coordinate_to_edge(self, i, j, box_length):
        """ Return format is tuple (box_id, edge) """
        # Horizontal line on the top
        if i % 2 == 0:
            if i == 0:
                return (int((j - 1) / 2), 'top_edge')
        # Horizontal line anywhere else
            else:
                return (int(box_length * (i / 2 - 1) + (j - 1) / 2), 'bottom_edge')
        # Vertical line on the left row
        if i % 2 == 1:
            if j == 0:
                return (int(box_length * (i - 1) / 2), 'left_edge')
            # Vertical line anywhere else
            else:
                return (int(box_length * (i - 1) / 2 + (j / 2) - 1), 'right_edge')

    def _cutoff_test(self, node_state, depth):
        """ Uses ply limit (aka horizon aka depth) to determine when to stop and evaluate the utility up to that point. """
        if (depth <= 0) or (self._term_test(node_state)):
            return True
        else:
            return False
    
    def _eval(self, node_state, horizon):
        """ A modified utility function.  A payoff function. """
        # We want the current AI score copy to make the util val higher in the differene, 
        # thus ranking it higher in the PQ 
        ai_score = copy.deepcopy(self.player_min_score)
        for box in node_state:
            if not box['box_closed']:
                if self._is_box_filled(node_state, box['box_id']):
                    ai_score += box['weight']

                    # We have a box with a single edge after 1 ply deep.
                    # Only check if the global ply limit is more than 1 
                    # to not affect results for games with only 1 ply limit depth. 
                    if horizon == self.ply_limit and self.ply_limit > 1:
                        # ranks it higher in the PQ
                        ai_score += 5

        return self.player_max_score - ai_score
        
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
            initial_state.append(box)

        return initial_state

    def _get_AI_move(self, node_state):
        """ Computates a move, then makes the move it sees best. """
        state = copy.deepcopy(node_state)
        move = self.MiniMax_Decision(state)
        print("\nAI's move is box {}, edge: {}\n".format(move[0], move[1]))
        self._set_edge(move[0], move[1])
        return

    def _get_edge_val(self, box, edge):
        """ Returns the value within the box's edge selected. """
        edge_value = self.state[box]['edges'][edge]
        return edge_value

    def _get_human_move(self, state):
        # Read human's move input
        value_in = input("\nChoose the coordinate of the edge you'd like to insert.\n" +
            "The first integer should be the row number, 2nd integer should be column. \nSeparate by space. Example: 0 1\nWill return the top edge of the first box. \n")
        
        inputs = value_in.split()
        inputs[0] = int(inputs[0])
        inputs[1] = int(inputs[1])
        box, edge = self._coordinate_to_edge(inputs[0], inputs[1], self.box_length)

        print("\nYour input was for box {}, edge {}\n".format(box, edge))

        # If it is legal, make that move (change assigned edge state)
        self._set_edge(box, edge)

    def _get_random_num(self):
        return random.randint(1, 5)
        
    def _get_random_move(self, state_length, legal_moves):
        action = []
        sum_of_edges = 0
        for box in legal_moves:
            for edge in box['edges']:
                sum_of_edges += 1
        # 3x3 board
        if state_length == 9 and len(legal_moves) > 8:
            if len(legal_moves[0]['edges']) > 3:
                action.append(0)
                action.append(legal_moves[0]['edges'].pop())
            elif len(legal_moves[8]['edges']) > 3:
                action.append(8)
                action.append(legal_moves[8]['edges'].pop())
            elif sum_of_edges > 27:
                num = self._get_random_num()
                action.append(num)
                action.append(legal_moves[num]['edges'].pop())
        # 4x4 board
        elif state_length == 16 and len(legal_moves) > 15:
            if len(legal_moves[0]['edges']) > 2:
                action.append(0)
                action.append(legal_moves[0]['edges'].pop())
            elif len(legal_moves[15]['edges']) > 2:
                action.append(15)
                action.append(legal_moves[15]['edges'].pop())
            elif sum_of_edges > 48:
                num = self._get_random_num()
                action.append(num)
                action.append(legal_moves[num]['edges'].pop())
        # 5x5 board
        elif state_length == 25 and len(legal_moves) > 24:
            if len(legal_moves[0]['edges']) > 2:
                action.append(0)
                action.append(legal_moves[0]['edges'].pop())
            elif len(legal_moves[24]['edges']) > 2:
                action.append(24)
                action.append(legal_moves[24]['edges'].pop())
            elif sum_of_edges > 75:
                num = self._get_random_num()
                action.append(num)
                action.append(legal_moves[num]['edges'].pop())

        return action

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

    def _get_edge_val_copy(self, node_state, box, edge):
        """ Returns the value within the box's edge selected. Box id will always be same as their index in the state array """
        edge_value = node_state[box]['edges'][edge]
        return edge_value

    def _is_box_filled(self, node_state, box):
        """ Returns true if all the edges in a box are filled. """
        if (node_state[box]['edges']['top_edge'] == 1) & (node_state[box]['edges']['bottom_edge'] == 1) & (node_state[box]['edges']['left_edge'] == 1) & (node_state[box]['edges']['right_edge'] == 1):
            return True
        else:
            return False

    def _is_double_edge_copy(self, node, box, edge):
        """ 
        Checks adjacent edges in the middle of the board 
        Returns True if the edges touch an adjacent box; returns False otherwise.
        """
        result = True
        # Another way to get sqrt of total boxes on board
        s = int(len(node['state']) ** 0.5)
        # 1st row
        if box < s and edge == 'top_edge':
            result = False
        # Last row
        if box >= s * (s-1) and edge == 'bottom_edge':
            result = False
        # Left column
        if box % s == 0 and edge == 'left_edge':
            result = False
        # Right column
        if (box + 1) % s == 0 and edge == 'right_edge':
            result = False
        return result

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

    def _make_node(self):
        ''' Creates state node '''
        node = {
            'state': copy.deepcopy(self.state),
            'path': 0
        }
        return node

    def _make_child_node(self, node_state, parent_node, action):
        """ (hopefully) Returns a COPY of the original node as to not affect the global state. """
        node = {
            'state': self._set_edge_copy(node_state, action),
            'parent_node': parent_node, 
            'action': action,
            'expanded': True 
        }
        return node

    def minimax(self, node, depth, maxPlayer):
        node_state = copy.deepcopy(node['state'])
        
        if self._cutoff_test(node_state, depth):
            return self._eval(node_state, depth)

        if maxPlayer:
            max_v = -9999
            legal_moves_per_box = self._get_possible_moves(node_state)
            for box in legal_moves_per_box:
                for edge in box['edges']:
                    action = [box['box_id'], edge]
                    child = self._make_child_node(node_state, node, action)
                    v = self.minimax(child, depth - 1, False)
                    max_v = max(max_v, v)
            return max_v
        else:
            min_v = 9999
            legal_moves_per_box = self._get_possible_moves(node_state)
            for box in legal_moves_per_box:
                for edge in box['edges']:
                    action = [box['box_id'], edge]
                    child = self._make_child_node(node_state, node, action)
                    v = self.minimax(child, depth - 1, True)
                    min_v = min(min_v, v)
            return min_v

    def minimax_alpha_beta(self, node, depth, maxPlayer, alpha, beta):
        node_state = copy.deepcopy(node['state'])
        
        if self._cutoff_test(node_state, depth):
            return self._eval(node_state, depth)

        if maxPlayer:
            max_v = -9999
            legal_moves_per_box = self._get_possible_moves(node_state)
            for box in legal_moves_per_box:
                for edge in box['edges']:
                    action = [box['box_id'], edge]
                    child = self._make_child_node(node_state, node, action)
                    v = self.minimax_alpha_beta(child, depth - 1, False, alpha, beta)
                    max_v = max(max_v, v)
                    alpha = max(alpha, v)
                    if beta <= alpha:
                        break
            return max_v
        else:
            min_v = 9999
            legal_moves_per_box = self._get_possible_moves(node_state)
            for box in legal_moves_per_box:
                for edge in box['edges']:
                    action = [box['box_id'], edge]
                    child = self._make_child_node(node_state, node, action)
                    v = self.minimax_alpha_beta(child, depth - 1, True, alpha, beta)
                    min_v = min(min_v, v)
                    beta = min(beta, v)
                    if beta <= alpha:
                        break
            return min_v
    
    def MiniMax_Decision(self, state):
        """ Returns an action """
        # Compute the element a of set S that has a maximum value of f(a)
        node_state = copy.deepcopy(state)
        legal_moves = self._get_possible_moves(node_state)

        # To reduce time it takes to make moves in the beginning of the game:
        if self.ply_limit >= 2 and len(legal_moves) > 8:
            random_move = self._get_random_move(len(self.state), legal_moves)
            if random_move != []:
                return random_move

        # default returns min at front. To get max, mult val by -
        pq = []
        # Tie-breaker counter. 
        pq_c = 0

        for box in legal_moves:
            boxID = box['box_id']

            # For each action available, return the utility value 
            for edge in box['edges']:
                depth = copy.deepcopy(self.ply_limit)
                action = [boxID, edge]
                child_node = self._make_child_node(node_state, state, action)
                t_val = self.minimax_alpha_beta(child_node, depth, False, -9999, 9999)
                heapq.heappush(pq, (t_val, pq_c, action))
                pq_c += 1

        # Return action with minimum util val
        action = heapq.heappop(pq) 
        action = action[2]
        return action

    def Play_game(self):
        """ Plays the game until there is a winner or human terminates game. """

        while(self.game_on):        
            node = self._make_node()
            node_state = copy.deepcopy(node['state'])
  
            self._print_board_state()
            
            if self._player():
                # Human's turn
                self._get_human_move(node_state)
            else:
                # AI's turn
                self._get_AI_move(node_state)
                
            self._terminal_test(self.state)
            if(self.game_on):
                self._player_switch()

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
            print("\nAI's turn now.")
        else:
            self.player = 1
            print("Human's turn now.")

    def _print_board_state(self):
        box_length = int(len(self.state) ** 0.5)
        print('   ', end='')
        for i in range(2 * box_length + 1):
            print(i, end=' ')
        print()
        print('  ', end='')
        for i in range(12 * box_length):
            print('_', end='')
        print()
        for i in range(2 * box_length + 1):
            print(i, end='| ')
            for j in range(2 * box_length + 1):
                # *'s
                if i % 2 == 0 and j % 2 == 0:
                    print('* ', end='')
                # Horizontal lines on the top row
                elif i % 2 == 0 and j % 2 == 1 and i == 0:
                    box_index = int((j - 1) / 2)
                    filled_edge = bool(self.state[box_index]['edges']['top_edge'])
                    if filled_edge:
                        print('- ', end='')
                    else:
                        print('  ', end='')
                # Horizontal lines everywhere else
                elif i % 2 == 0 and j % 2 == 1 and i != 0:
                    box_index = int(box_length * ((i / 2) - 1) + (j - 1) / 2)
                    filled_edge = bool(self.state[box_index]['edges']['bottom_edge'])
                    if filled_edge:
                        print('- ', end='')
                    else:
                        print('  ', end='')
                # Vertical lines on the left row
                elif i % 2 == 1 and j % 2 == 0 and j == 0:
                    box_index = int((i - 1) / 2) * box_length
                    filled_edge = bool(self.state[box_index]['edges']['left_edge'])
                    if filled_edge:
                        print('| ', end='')
                    else:
                        print('  ', end='')
                # Vertical lines everywhere else
                elif i % 2 == 1 and j % 2 == 0 and j != 0:
                    box_index = int((i - 1) / 2 * box_length + (j - 1) / 2)
                    filled_edge = bool(self.state[box_index]['edges']['right_edge'])
                    if filled_edge:
                        print('| ', end='')
                    else:
                        print('  ', end='')

                else:
                    if (i + j) % 2 == 0:
                        box_index = int(box_length * (i - 1) / 2 + (j - 1) / 2)
                        print(self.state[box_index]['weight'], end=' ')
            print()
    
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

    def _set_edge(self, box, edge):
        """ Flip edge state from 0 (available) to 1 (filled) """
        if not self._get_edge_val(box, edge):
            self.state[box]['edges'][edge] = 1
            # Check if box filled
            self._box_filled(box)
            # print("New state of box: {}".format(self.state[box]))
            # Check adjacent edges
            self._set_adj_edges(box, edge)
        else:
            # This is also a way for _set_adj_edges & _set_edge to break out of an infinite loop 
            return
        
    def _set_edge_copy(self, node_state, action):
        """ Flip edge state from 0 (available) to 1 (filled) """
        box = action[0]
        edge = action[1]
        if not self._get_edge_val_copy(node_state, box, edge):
            new_state = copy.deepcopy(node_state)
            new_state[box]['edges'][edge] = 1
            return new_state
        else:
            return node_state

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
        
    def _term_test(self, node_state):
        """ Terminates a cutoff search if a box has been filled. """
        is_box_closed = False
        for box in node_state:
            if not box['box_closed']:
                if self._is_box_filled(node_state, box['box_id']):
                    is_box_closed = True
        return is_box_closed


twoByTwo = 4   # boxes
threeByThree = 9   # boxes
fourByFour = 16   # boxes

game = BoxGame(twoByTwo, 2)
game.Play_game()
