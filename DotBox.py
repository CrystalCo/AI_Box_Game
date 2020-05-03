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
        self.random_corner_edge_dict = [
            [len(self.state) - 1, 'right_edge'], 
            [len(self.state) - 1, 'bottom_edge'], 
            [0, 'top_edge'], 
            [0, 'left_edge']
        ]
    
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

    def _box_filled(self, node, box):
        """ 'Closes' box by changing box_close to True if box is filled. """
        if self._is_box_filled(node, box):
            node['state'][box]['box_closed'] = True
            new_node = self._add_points_copy(node, box)
            return new_node 
        else:
            return node

    def _is_box_filled(self, node, box):
        """ Returns true if all the edges in a box are filled. """
        if (node['state'][box]['edges']['top_edge'] == 1) & (node['state'][box]['edges']['bottom_edge'] == 1) & (node['state'][box]['edges']['left_edge'] == 1) & (node['state'][box]['edges']['right_edge'] == 1):
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

    def _add_points(self, node, box):
        """ Adds points to player """
        if node['player']:
            node['player_max_score'] += node['state'][box]['weight']
            print("Human scores.  Human total: {}. AI total: {}.\n".format(node['player_max_score'], node['player_min_score']))
            return node
        else:
            node['player_min_score'] += node['state'][box]['weight']
            print("AI scores.  Human total: {}. AI total: {}.\n".format(node['player_max_score'], node['player_min_score']))
            return node
   
    def _set_edge(self, node_state, box, edge):
        if not self._get_edge_val_copy(node_state, box, edge):
            new_state = copy.deepcopy(node_state)
            new_state[box]['edges'][edge] = 1
            return new_state
        else:
            return node_state

    def _get_edge_val(self, node_state, box, edge):
        """ Returns the value within the box's edge selected. Box id will always be same as their index in the state array """
        edge_value = node_state[box]['edges'][edge]
        return edge_value

    def _set_adj_edges(self, node, box, edge):
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

    def _is_double_edge(self, node, box, edge):
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
 
    def _result(self, state, action):
        """
            The transition model, which defines the result of a move.
            Node must include the state.  
            Action must be an array with the box id as the first element & edge as the 2nd element
            Returns a new state.
        """

    def _make_node(self):
        ''' Creates state node '''
        node = {
            'state': copy.deepcopy(self.state),
            'depth': 0
        }
        return node

    def _make_child_node(self, node_state, parent_node, action):
        """ (hopefully) Returns a COPY of the original node as to not affect the global state. """
        node = {
            'state': node_state,
            'parent_node': parent_node, # needed?
            'action': action,
            # 'depth': depth, # needed here?
            'player': parent_node['player'],    # if only AI is using it, is this needed?
            'max_score': parent_node['player_max_score'],
            'min_score': parent_node['player_min_score'],
            'expanded': True    # needed?
        }
        return node

    def _Play_Game(self):
        # Initiate a node with a copy of the state
        node = self._make_node()

        while(self.game_on):
            if self._player():
                # Get Human's move
                self._get_human_move()

                # Save the result
                # Get AI's move
                    # Pass AI a copy of the state
                    # AI 
                # Set the result

    def _player(self):
        """ 
            Defines which player has the move in a state. 
            True  = Human;      False = AI
            Player max = human;   Player min = AI 
        """
        return self.player
        
easy_2x2 = 4   # boxes

game = BoxGame(easy_2x2, 1)
game.Play_game()

