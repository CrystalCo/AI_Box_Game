# Crystal Contreras     SPRING 2020 CSC-480
import random

class BoxGame:
    # When the game starts the human player will specify the board size and ply limit
    def __init__(self, state, ply_limit):
        # The size of the board
        self.h_edges = 0 
        self.v_edges = 0 
        self.vertices = 0
        # the state of the board with randomly generated weights
        self.state = self._generate_board(state)
        # how many plys the AI will search (i.e., the horizon for the minimax).
        self.ply_limit = ply_limit
        self.player_min = 0
        self.player_max = 0
    
    def _generate_board(self, boxes):
        """ Initializes the board with edges, vertices, & weights.
        Returns weights for each box on the board 
        """
        weights = []
        for i in range(boxes**2):
            weights.append(random.randint(1, 5))
        print("weights: ", weights)
        self._create_edges(boxes)
        return weights

    def _create_edges(self, boxes):
        """ Calculates the number of edges on the board"""
        num_same_side_edge = boxes * (boxes + 1)
        edges = 2 * num_same_side_edge
        # Initialize 2 empty stacks
        horizontal_edges = []
        vertical_edges = []
        for i in range(num_same_side_edge):
            horizontal_edges.append('_')
            vertical_edges.append('|')
        self.h_edges = horizontal_edges
        self.v_edges = vertical_edges
        print("Edges: {} {}".format(horizontal_edges, vertical_edges))
        return edges

    def _create_vertices(self, box):
        """ Needed? """

    def _player(self, state):
        """ Defines which player has the move in a state. """

    def _actions(self, state):
        """Returns the set of legal moves in a state."""
    
    def _result(self, state, action):
        """The transition model, which defines the result of a move"""

    def _terminal_test(self, state):
        """ A terminal test, which is true when the game is over and false otherwise. 
        States where the game has ended are called terminal states.
        Terminal state is when all edges are filled/popped.
        """
    
    def _utility(self, state, player):
        """A payoff function.  Defines the final numeric value for a game 
        that ends in terminal state s for a player p. 
        Outcome is win, loss, or draw (+1, 0, 1/2)
        """

        
easy_2x2 = 2   # boxes
# Let's say we want a 2x2 board.  That would be 12 edges/9 vertices/4 weights:
#  weights:  2^2=4
#  edges:   4*3 || 2*3 + 2*3 = 12
#  _ _      * - * - *
# |_|_|     |   |   |
# |_|_|     * _ * _ *
#           |   |   |
#           * _ * _ *

# 3x3 board = 12 + 12 = 24 edges/16 vertices/ 9 weights 
# weights:  3^2 = 9
# edges:    3*4 + 3*4 = 24
# easy_3x3 = 9 boxes
#  _ _ _
# |_|_|_|
# |_|_|_|
# |_|_|_|

game = BoxGame(easy_2x2, 2)
game
