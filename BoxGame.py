# Crystal Contreras     SPRING 2020 CSC-480
import random
import math
import heapq
import numpy as np
import copy

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
        print("\nLegal Moves:\n{}\n".format(legal_moves))
        
        # Read human's move input
        value_in = input("Enter the box id and edge from the legal moves returned. Separate answer with a space.\nExample:   2 top_edge \n")
        inputs = value_in.split()
        box = int(inputs[0])
        edge = inputs[1]

        print("\nYour input was: {} {}\n".format(box, edge))

        # If it is legal, make that move (change assigned edge state)
        self._set_edge(box, edge)
        
    def _get_AI_move(self, node_state):
        """ Computates a move, then makes the move it sees best. """
        state = copy.deepcopy(node_state)
        move = self.MiniMax_Decision(state)
        print("AI move: ", move)
        self._set_edge(move[0], move[1])
        return

    def MiniMax_Decision_2(self, node):
        """ Returns an action array that contains the box_id and edge. """
        node_state = node['state']

        # Compute the element a of set S that has a maximum value of f(a)
        legal_moves = self._get_possible_moves(node_state)
        box1 = legal_moves[0]
        print("\nLegal Moves:\n{}\n".format(box1))

        # default returns min at front. To get max, mult val by -
        pq = []
        # Tie-breaker counter. 
        pq_c = 0

        # for box in legal_moves:
            # For each action available, return the utility value 
        for edge in box1['edges']:
            depth = self.ply_limit
            action = [box1['box_id'], edge]
            child = self._result(node, action)
            print("New child: {}".format(child))
            t_val = self.minimax(child, depth, True)
            heapq.heappush(pq, (t_val * -1, pq_c, action))
            pq_c += 1

        # Return max
        action = heapq.heappop(pq) 
        action = action[2]
        return action

    def minimax(self, node, depth, maximizingPlayer):
        node_state = copy.deepcopy(node['state'])
        
        if self._cutoff_test(node_state, depth):
            return self._eval(node_state)

        if maximizingPlayer:
            print("Max Player fn.")
            max_v = -9999
            legal_moves_per_box = self._get_possible_moves(node_state)
            # first_box = legal_moves_per_box[0]
            for box in legal_moves_per_box:
                for edge in box['edges']:
                    action = [box['box_id'], edge]
                    # child = self._result(node, action)
                    child = self._make_child_node(node_state, node, action)
                    v = self.minimax(child, depth - 1, False)
                    max_v = max(max_v, v)
            return max_v
        else:
            print("Min player fn.")
            min_v = 9999
            legal_moves_per_box = self._get_possible_moves(node_state)
            # first_box = legal_moves_per_box[0]
            for box in legal_moves_per_box:
                for edge in box['edges']:
                    action = [box['box_id'], edge]
                    # child = self._result(node, action)
                    child = self._make_child_node(node_state, node, action)
                    v = self.minimax(child, depth - 1, True)
                    min_v = min(min_v, v)
            return min_v

    def MiniMax_Decision(self, state):
        """ Returns an action """
        # Compute the element a of set S that has a maximum value of f(a)
        node_state = copy.deepcopy(state)
        legal_moves = self._get_possible_moves(node_state)
        print("\nLegal Moves:\n{}\n".format(legal_moves))

        # Rank boxes that only have 1 edge left as top priorities.
        # This gives us a chance to reduce the amount of checks we do below.
        # Remove once algorithm is complete
        # for m in legal_moves:
        #     if len(m['edges']) == 1:
        #         action = [m['box_id'], m['edges'].pop()]
        #         return action

        # default returns min at front. To get max, mult val by -
        pq = []
        # Tie-breaker counter. 
        pq_c = 0

        for box in legal_moves:
            print("Box: ", box)
            boxID = box['box_id']

            # TODO: Implement an "explored" set to remove redundant edges
    
            # For each action available, return the utility value 
            for edge in box['edges']:
                depth = copy.deepcopy(self.ply_limit)
                action = [boxID, edge]
                child_node = self._make_child_node(node_state, state, action)

                t_val = self.minimax(child_node, depth, True)
                # heapq.heappush(pq, (t_val * -1, pq_c, action))    Multiplied by a negative number so that it would turn positive numbers into negative ones, thus making it higher in the PQ since this PQ is a min heap
                heapq.heappush(pq, (t_val, pq_c, action))
                pq_c += 1

        # Return max
        action = heapq.heappop(pq) 
        action = action[2]
        return action

    def _get_possible_moves_per_box(self, state):
        '''  Returns a list of available edges. '''
        available_edges = []
        # Need a counter for index of available edges since boxes that are closed will be skipped
        if state['box_closed'] == False:
            available_edges.append({
                'box_id': state['box_id'],
                'weight': state['weight'],
                'edges': [] 
            })

            for e, val in state['edges'].items():
                if val == 0:
                    available_edges[0]['edges'].append(e)

        return available_edges

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

    def _box_filled_copy(self, node, box):
        """ 'Closes' box by changing box_close to True if box is filled. """
        if self._is_box_filled(node['state'], box):
            node['state'][box]['box_closed'] = True
            new_node = self._add_points_copy(node, box)
            return new_node 
        else:
            return node

    def _is_box_filled(self, node_state, box):
        """ Returns true if all the edges in a box are filled. """
        if (node_state[box]['edges']['top_edge'] == 1) & (node_state[box]['edges']['bottom_edge'] == 1) & (node_state[box]['edges']['left_edge'] == 1) & (node_state[box]['edges']['right_edge'] == 1):
            return True
        else:
            return False

    def _adj_box_filled(self, node):
        """ Check if marking off an adjacent box resulted in its neighbor's box getting filled. """
        new_node = node
        for box in node['state']:
            if not box['box_closed']:
                if (box['edges']['top_edge'] == 1) & (box['edges']['bottom_edge'] == 1) & (box['edges']['left_edge'] == 1) & (box['edges']['right_edge'] == 1):
                    node['state'][box['box_id']]['box_closed'] = True
                    new_node = self._add_points_copy(node, box['box_id'])
                    return new_node 
        # else no adjacent boxes were filled. Can return original node
        return new_node

    def _add_points(self, box):
        """ Adds points to player """
        if self._player():
            self.player_max_score += self.state[box]['weight']
            print("Human scores.  Human total: {}. AI total: {}.\n".format(self.player_max_score, self.player_min_score))
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

    def _set_edge(self, box, edge):
        """ Flip edge state from 0 (available) to 1 (filled) """
        if not self._get_edge_val(box, edge):
            self.state[box]['edges'][edge] = 1
            # Check if box filled
            self._box_filled(box)
            print("New state of box: {}\n".format(self.state[box]))
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

    def _get_edge_val_copy(self, node_state, box, edge):
        """ Returns the value within the box's edge selected. Box id will always be same as their index in the state array """
        edge_value = node_state[box]['edges'][edge]
        return edge_value

    def _set_adj_edges_copy(self, node, box, edge):
        """ Marks off adjacent edges """
        boxes = int(len(node['state']))
        if boxes == 1:
            # single box board doesn't have adjacent boxes to check 
            return node
        if self._is_double_edge_copy(node, box, edge):
            new_box, new_edge = self.adj_edge_dict[edge]
            new_box += box
            new_node = self._set_edge_copy(node, new_box, new_edge)
            return new_node
        else:
            return node

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
 
    def _result(self, node, action):
        """
            The transition model, which defines the result of a move.
            Node must include the state.  
            Action must be an array with the box id as the first element & edge as the 2nd element
            Returns a new state.
        """
        # Returns a new state
        node_state = copy.deepcopy(node['state'])
        new_state = self._set_edge_copy(node_state, action, action)
        # new_node = self._set_edge_copy(node, action[0], action[1])

        # Create child node copy
        child = self._make_child_node(new_state, node, action)

        # Check if box filled.  Returns a new score
        child = self._box_filled_copy(child, action[0])
        print("New state of child box: {}\n".format(child['state'][action[0]]))
        
        # Checks adjacent edges.  Returns a new state if adj edges are marked
        child = self._set_adj_edges_copy(child, action[0], action[1])

        # Check if adj box was filled.  Returns a new score if so
        new_child = self._adj_box_filled(child)

        # If adjacent edges were found, print new state of that box
        if new_child != child:
            print("New state of box after checking adjacent edges:    {}\n".format(new_child['state'][action[0]]))
    
        return new_child

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

    def Play_game(self):
        """ Plays the game until there is a winner or human terminates game. """

        while(self.game_on):        
            node = self._make_node()
            node_state = copy.deepcopy(node['state'])
            
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
            print("AI's turn now.\n")
        else:
            self.player = 1
            print("Human's turn now.\n")
   
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
        
    def _cutoff_test(self, node_state, depth):
        """ Uses ply limit (aka horizon aka depth) to determine when to stop and evaluate the utility up to that point. """
        if (depth <= 0) or (self._term_test(node_state)):
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

    def _eval(self, node_state):
        ai_score = copy.deepcopy(self.player_min_score)
        # print("AI Score eval'ed: ", ai_score)
        for box in node_state:
            if not box['box_closed']:
                if self._is_box_filled(node_state, box['box_id']):
                    ai_score += box['weight']
                    print("Score eval'ed: ", ai_score)
        return self.player_max_score - ai_score

    def _evaluation(self, node):
        """ A modified utility function.  A payoff function. """
        return node['player_max_score'] - node['player_min_score']
    # def H_MiniMax(self, state, depth):
    #     if self._cutoff_test(state, depth):
    #         return self._evaluation(state)
        
    #     if self._player(state):
    #         array = 
    #         return max([2*k for k in some_array])



"""
the suggestion is to alter minimax in two ways: 

1. Replace the utility function by a heuristic evaluation function EVAL, which estimates the position’s utility; and
2. Replace the terminal test by a cutoff test that decides when to apply EVAL.

"""
        
easy_2x2 = 4   # boxes

game = BoxGame(easy_2x2, 1)
game.Play_game()

