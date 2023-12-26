### montecarlorl.py
#!/usr/bin/env python3

import os
import sys
from typing import List
import random
import numpy as np
import time
from math import sqrt, log

# Since I am using non standard directories for grouping classes, add them to path
from os.path import dirname
sys.path.append(os.path.join(dirname(__file__), r'../Cut'))
from graph import Graph, Node, Edge
from tictactoe import hash, findUniqueEdges, gameOver, winner, findEmptyCells, getHash


class Value:
    """
    The Value class represents the value of a game state.
    It is calculated as the average value of its children, so this class
    determines the average of all values provided during updates.
    """
    def __init__(self):
        """
        Initialise the Value to 0.0, and the number of floats it has averaged as 0

        Time Complexity: Theta(1)
        """
        self.value = 0.0
        self.n = 0

    def update(self, value: float):
        """
        Update the average using the provided new float

        Updating the average in this way avoids overflow
        compared with storing the total itself

        Time Complexity: Theta(1)
        """
        self.n += 1
        self.value += (value - self.value) / float(self.n)


def createRandomPolicy(choices: int) -> List[float]:
    """
    Initialise the Policy (actually the Q values providing a list of probabilities for each choice)
    so that each has equal weight

    Time Complexity: Theta(N)
    """
    if choices > 0:
        p = 1.0 / float(choices)
        l = [p for x in range(choices)]
        return l
    else:
        return []
    
def createZeroPolicy(choices: int) -> List[float]:
    """
    Initialise the Policy (actually the Q values providing a list of probabilities for each choice)
    so that each has zero weight

    Time Complexity: Theta(N)
    """
    return [0.0 for x in range(choices)]


def printBoard(board, board_size):
    """
    Print the board to the console.

    Parameters:
    board: A list of values (0 = empty, 1 or 2 are player numbers)
    board_size: The side length of the board

    Returns: Nothing 

    Time Complexity: Theta(N)
    """
    for y in range(board_size):
        for x in range(board_size):
            sys.stdout.write(f" {board[x + y * board_size]} ")
            sys.stdout.write(f"|" if x < board_size - 1 else "\n")
        if y < board_size -1:
            for x in range(board_size):
                sys.stdout.write("---")
                sys.stdout.write("*" if x < board_size - 1 else "\n")
        else:
            sys.stdout.write("\n")


def getRandomState(boardSize, numPlayers):
    """
    Return a randomly selected state for a game where each possible state has a non-zero chance of selection
    Because the hashes of the legal boards are a very small proportion of the numPlayers^(boardSize^2), randomly
    selecting hashes is not effective.

    Instead play through a game with random selection, recording the states encountered
    Then return one of them chosen at random.

    Parameters:
    boardSize: int - side length of the board
    numPlayers: int - number of players in the game

    Returns:
    board: List[int] selected at random
    boardHash: The hash of the board

    Time Complexity: O(N^3)
    as it simulates up to N moves, calling find unique edges which is N^2
    where N is the number of spaces on the board (eg 25 for a 5x5 board)
    """
    board = [0] * (boardSize * boardSize)
    board, boardHash = hash(board, boardSize, numPlayers)

    boards = [board]
    states = [boardHash]
    player = 1

    # Randomly play through a game, storing the states in a list
    # This gives a non-zero probability of reaching each state
    while not gameOver(board, boardSize):
        newResult, newHashes, newIndices = findUniqueEdges(board, boardSize, player, numPlayers)
        action = random.randrange(len(newIndices))
        board = newResult[action]
        boardHash = newHashes[action]
        player = (player % numPlayers) + 1
        if player == 1:
            states.append(boardHash)
            boards.append(board)

    # Now select a random state from the playthough and return it
    index = random.randrange(len(states))
    return boards[index], states[index]


class Model:
    def __init__(self):
        """
        Initialise a Model with an empty graph and no policies or values calculated

        Time Complexity: Theta(1)
        """
        self.graph = Graph()
        self.policy = {}
        self.value = {}
        self.numPlayers = None
        self.boardSize = None
        self.epochs = 0

    def save(self, filename: str):
        """
        Save the Model to disk

        Existing Python libraries produced memory overflows with large models,
        so the JSON is output manually

        Params:
        filename: String - The file to create/overwrite with the model

        Returns:
        Nothing

        Time Complexity: O(V+E)
        """
        with open(filename, "w") as f:
            f.write("{")
            f.write(f'"numPlayers": {self.numPlayers}')
            f.write(f', "boardSize": {self.boardSize}')
            f.write(f', "epochs": {self.epochs}')
            f.write(f', "graph": {{')
            f.write(f'Vertices: [')
            for i, v in enumerate(self.graph.vertices):
                if i > 0:
                    f.write(f', ')
                f.write(f'{v.value}')
            f.write(f'], Edges: [')
            i = 0
            for v in self.graph.vertices:
                for e in v.edges:
                    if i > 0:
                        f.write(f', ')
                    f.write(f'{{"u": {e.u}, "v": {e.v}, "action": {e.action}, "index": {e.index}, "value": ["value": {e.value.value}, "n": {e.value.n}]}}')
                    i += 1
            f.write(f']')
            f.write(f'}}')
            f.write(f', "policy": {{')
            for i, (k, v) in enumerate(self.policy.items()):
                if i > 0:
                    f.write(f', ')
                f.write(f'{k}: {v}')
            f.write(f'}}')
            f.write(f', "value": {{')
            for i, (k, v) in enumerate(self.value.items()):
                if i > 0:
                    f.write(f', ')
                f.write(f'{k}: {{"value": {v.value}, "n": {v.n}}}')
            f.write(f'}}')
            f.write("}")


    def fit_montecarlo(self, numPlayers, boardSize, learning_rate, epochs):
        """
        Train the model using a Monte Carlo Reinforcement Learning approach with Exploring Starts

        The model is trained against an opponent which uses a random policy.
        It uses AfterStates so the resulting board after the first player's move
        is the input state for the next player

        If the model has already been trained, it must be called with the same number of players and board size.
        Training will be continued for the given number of epochs.

        Model-free Monte-Carlo: 
        not attempt to learn the transition dynamics (probabilities) or 
        the reward structure of the Markov Decision Process (MDP). 
        Instead, It learns the value for each state which is stored in the nodes, and
        also calculates Q(S, a) which is stored in each edge in the Graph.

        Parameters:
        numPlayers: int - the number of players in the game
        boardSize: int - the side length of the board
        learningRate: float - the rate of decay of the reward at each step
        epochs: int - the number of completed games to simulate during training

        Time Complexity: O(Epochs * N^3)
          as it simulates epochs games of up to N moves, calling FindUniqueEdges which is O(N^2)
        """

        # Validate arguments
        if numPlayers < 1:
            raise RuntimeError(f"There must be at least one player")
        if self.numPlayers == None:
            self.numPlayers = numPlayers
        else:
            if self.numPlayers != numPlayers:
                raise("Number of players can not change between calls to fit")
            
        if boardSize < 3:
            raise RuntimeError(f"The minimum board size is 3x3")
        if self.boardSize == None:
            self.boardSize = boardSize
        else:
            if self.boardSize != boardSize:
                raise("Board size can not change between calls to fit")

        if epochs < 0:
            raise RuntimeError("The number or epochs must be positive")        
        self.epochs += epochs

        # Ordinarily the algorithm would initialise Policy and Values for all states
        # but this number grows rapidly with board size, and would be impractical if only
        # a limited number of states will be visited by the algorithm. Instead I use lazy initialisation
        # so all states not already stored will be initialised with a uniform random policy
        # and unknown value when it is first encountered.
        for _ in range(epochs):
            # Exploring starts:
            # Begin in a randomly selected state (where all states have a non-zero probability of selection)
            # This roughly doubles the runtime for each playthrough (as it simulates a random game to find the state)
            # but is necessary to guarantee convergence of the policy (providing the learning rate is also small enough,
            # and enough episodes are also simulated)
            board, boardHash = getRandomState(boardSize, numPlayers)
            episode = []
            transitions = []
            episode.append(boardHash)

            while not gameOver(board, boardSize):
                # 2. Select the policy and value for this state
                if boardHash in self.graph.map: 
                    # if it already exists in the graph
                    current = self.graph.map[boardHash]
                    currentPolicy = self.policy[boardHash]
                    currentValue = self.value[boardHash]

                    # Determine possible moves for this board
                    result, hashes, indices = findUniqueEdges(current.board, boardSize, current.player, numPlayers)
                else:
                    # This node is not in the graph, add it
                    current = Node(boardHash)
                    current.board = board
                    current.player = 1
                    self.graph.addVertex(current)

                    # determine possible moves from this position
                    result, hashes, indices = findUniqueEdges(current.board, boardSize, current.player, numPlayers)

                     # Lazy initialisation of policy and value function for state boardHash
                    currentPolicy = createRandomPolicy(len(indices))
                    currentValue = Value()
                    self.policy[boardHash] = currentPolicy
                    self.value[boardHash] = currentValue

                # 3. action selection
                # The agent selects an action based on the policy for that state. 
                # For the first state in the episode, the action is chosen randomly (exploring starts). 
                # For subsequent states, the action is selected based on the current policy.
                if len(episode) == 1:
                    action = random.randrange(len(indices))
                else:
                    # index = np.argmax(currentPolicy)
                    action = np.random.choice(np.flatnonzero(currentPolicy == np.max(currentPolicy))) # Better than argmax is to randomly split ties

                # 4. Transition to the new state            
                newHash = hashes[action]
                newNode = Node(newHash)
                newPlayer = (current.player % numPlayers) + 1 
                newNode.player = newPlayer
                newNode.board = result[action]

                # Add the new state to the graph (if needed)
                if not newHash in self.graph.map:
                    self.graph.addVertex(newNode)

                    _, _, newIndices = findUniqueEdges(newNode.board, boardSize, newNode.player, numPlayers)
                    self.policy[newHash] = createRandomPolicy(len(newIndices))
                    self.value[newHash] = Value()


                # Add an edge between the current and new state (if needed)
                newEdge = current.findEdge(current, newNode)
                if newEdge is None:
                    newEdge = Edge(current, newNode, 0.0)
                    newEdge.action = indices[action]
                    newEdge.index = action
                    newEdge.value = Value()
                    current.edges.append(newEdge)
                transitions.append(newEdge)

                # Now simulate the opponent's turn
                # This follows a similar pattern to above, but uses a random policy from all states
                board = newNode.board
                boardHash = newHash
                current = newNode
                episode.append(boardHash)

                if not gameOver(board, boardSize):
                    result, hashes, indices = findUniqueEdges(current.board, boardSize, current.player, numPlayers)
                    action = random.randrange(len(indices))

                    newHash = hashes[action]
                    newNode = Node(newHash)
                    newPlayer = (current.player % numPlayers) + 1 
                    newNode.player = newPlayer
                    newNode.board = result[action]

                    # Add the new state to the graph (if needed)
                    if not newHash in self.graph.map:
                        self.graph.addVertex(newNode)

                        _, _, newIndices = findUniqueEdges(newNode.board, boardSize, newNode.player, numPlayers)
                        self.policy[newHash] = createRandomPolicy(len(newIndices))
                        self.value[newHash] = Value()

                    # Add an edge between the current and new state (if needed)
                    newEdge = current.findEdge(current, newNode)
                    if newEdge is None:
                        newEdge = Edge(current, newNode, 0.0)
                        newEdge.action = indices[action]
                        newEdge.index = action
                        newEdge.value = Value()
                        current.edges.append(newEdge)
                    transitions.append(newEdge)

                    # Prepare for the original player's turn
                    board = newNode.board
                    boardHash = newHash
                    current = newNode
                    episode.append(boardHash)

            # If we started from a terminal state, it might not have been added yet
            if not boardHash in self.graph.map:
                current = Node(boardHash)
                current.board = board
                current.player = 1
                self.graph.addVertex(current)

                # Lazy initialisation of policy and value function for state boardHash
                result, hashes, indices = findUniqueEdges(current.board, boardSize, current.player, numPlayers)

                currentPolicy = createRandomPolicy(len(indices))
                currentValue = Value()
                self.policy[boardHash] = currentPolicy
                self.value[boardHash] = currentValue

            # 4. Value Update
            win = winner(board, boardSize)
            reward = 1.0 if win == 1 else -1.0

            # Update value based on reward
            # It is not possible for a state to appear twice within the one episode because marks are only added
            # so simply add the discounted reward based on the learning rate
            # This means that first visit and every visit Monte Carlo search are the same for this problem
            # The terminal state gets the full reward.
            reward = 1.0 if win == 1 else -1.0
            for state in reversed(episode):
               self.value[state].update(reward)
               reward *= learning_rate

            
            # 5. Policy Update: The Q-values (stored in the edges of the graph) are updated based on the reward.
            reward = 1.0 if win == 1 else -1.0
            for edge in reversed(transitions):
                edge.value.update(reward)
                self.policy[edge.u.value][edge.index] = edge.value.value # The policy for the starting state of that edge (edge.u.value) is updated based on the Q-value (edge.value.value).
                reward *= learning_rate


    def fit_sarsa(self, numPlayers, boardSize, gamma, alpha, epsilon, epochs):
        """
        Train the model using SARSA

        SARSA provides on policy temporal difference control, estimating Q and updating the policy
        during the epoch.

        The model is trained against an opponent following a random policy.
        It does not use AfterStates but instead only treats states where the model can move as a state
        and the other players as part of the environment

        If the model has already been trained, it must be called with the same number of players and board size.
        Training will be continued for the given number of epochs.

        Parameters:
        numPlayers: int - the number of players in the game
        boardSize: int - the side length of the board
        gamma: float - the discount rate applied to rewards from the neighbouring state
        alpha: float - the step size to update the weights
        epsilon: float - the probability of exploring (rather than exploiting the Greedy policy)
        epochs: int - The number of completed games to simulate during training

        Return:
        Nothing

        Time Complexity: O(Epochs * N^3)
          as it simulates epochs games of up to N moves, calling FindUniqueEdges which is O(N^2)
        """

        # Validate arguments
        if numPlayers < 1:
            raise RuntimeError(f"There must be at least one player")
        if self.numPlayers == None:
            self.numPlayers = numPlayers
        else:
            if self.numPlayers != numPlayers:
                raise("Number of players can not change between calls to fit")
            
        if boardSize < 3:
            raise RuntimeError(f"The minimum board size is 3x3")
        if self.boardSize == None:
            self.boardSize = boardSize
        else:
            if self.boardSize != boardSize:
                raise("Board size can not change between calls to fit")
        
        if epochs < 0:
            raise RuntimeError("The number or epochs must be positive")        
        self.epochs += epochs

        if alpha <= 0.0 or alpha > 1.0:
            raise RuntimeError("Alpha should be in range (0, 1]")

        if gamma < 0.0 or gamma > 1.0:
            raise RuntimeError("Gamma should be in range [0, 1]")

        if epsilon <= 0.0 or epsilon > 1.0:
            raise RuntimeError("Epsilon should be small and in range (0, 1]")

        for _ in range(epochs):
            # Each playthrough starts from an empty board (rather than using exploring starts)
            board = [0] * (boardSize * boardSize)
            board, boardHash = hash(board, boardSize, numPlayers)
            currentPlayer = 1

            # Selecta a move with epsilon-Greedy strategy
            # This is different to MonteCarloRL with ES which always explores for the first turn
            # and then exploits all remaining states.
            _, _, indices = findUniqueEdges(board, boardSize, currentPlayer, numPlayers)
            if boardHash in self.policy.keys():
                currentPolicy = self.policy[boardHash]
                if random.random() < epsilon:
                    index = random.randrange(len(indices))
                else:
                    index = np.random.choice(np.flatnonzero(currentPolicy == np.max(currentPolicy)))
            else:
                # If this state has not been visited before, use lazy initialisation to set all weights to 0
                # SARSA requires arbitrary initialisation of all non-terminal states,
                # and terminal states should be initialised with Q(S, a) = 0 for all a
                currentPolicy = createZeroPolicy(len(indices))
                self.policy[boardHash] = currentPolicy
                index = random.randrange(len(indices))
            action = indices[index]
            
            while not gameOver(board, boardSize):
                # Process the selected move for the first player (the model)
                newBoard = board.copy()
                newBoard[action] = currentPlayer
                newBoard, newHash = hash(newBoard, boardSize, numPlayers)
                currentPlayer = (currentPlayer % numPlayers) + 1

                # Rather than using Afterstates, this model only considers states
                # from which the model can move as states, and environment represents the other player's actions
                # Take turns for all the other players (assumed to all follow a uniform random policy)
                while currentPlayer != 1 and not gameOver(newBoard, boardSize):
                    newAction = random.randrange(len(findEmptyCells(newBoard, boardSize)))
                    newBoard[newAction] = currentPlayer
                    newBoard, newHash = hash(newBoard, boardSize, numPlayers)
                    currentPlayer = (currentPlayer % numPlayers) + 1

                # Choose subsequent action for the model's next iteration
                if gameOver(newBoard, boardSize):
                    newPolicy = [0]
                    newIndex = 0
                    newAction = 0
                else:
                    _, _, newIndices = findUniqueEdges(newBoard, boardSize, currentPlayer, numPlayers)
                    if newHash in self.policy.keys():
                        newPolicy = self.policy[newHash]
                        if random.random() < epsilon:
                            newIndex = random.randrange(len(newIndices))
                        else:
                            newIndex = np.random.choice(np.flatnonzero(newPolicy == np.max(newPolicy)))
                    else:
                        newPolicy = createZeroPolicy(len(newIndices))
                        self.policy[newHash] = newPolicy
                        newIndex = random.randrange(len(newIndices))
                    newAction = newIndices[newIndex]
            
                # Update policy
                # Reward is +1 for a win, -1 for a loss or draw
                if gameOver(newBoard, boardSize):
                    win = winner(newBoard, boardSize)
                    reward = 1.0 if win == 1 else -1.0
                else:
                    reward = 0.0

                # Sarsa: Q(S, A) = Q(S, A) + alpha(R + gamma*Q(S', A') - Q(S, A))
                update = alpha * (reward + gamma * newPolicy[newIndex] - currentPolicy[index])
                self.policy[boardHash][index] += update

                # Prepare for next iteration
                board = newBoard
                boardHash = newHash
                index = newIndex
                action = newAction
                indices = newIndices
                currentPolicy = newPolicy


    def fit_qlearning(self, numPlayers, boardSize, gamma, alpha, epsilon, epochs):
        """
        Train the model using Q learning

        Q learning is off policy temporal difference control, estimating Q
        independently from the policy used as the maximum value Q(S, a) for any
        actions from each state is used in updates.

        However this also means more states are evaluated during each episode based on the branching factor

        The model is trained against an opponent following a random policy.
        It does not use AfterStates but instead only treats states where the model can move as a state
        and the other players as part of the environment

        If the model has already been trained, it must be called with the same number of players and board size.
        Training will be continued for the given number of epochs.

        Parameters:
        numPlayers: int - the number of players in the game
        boardSize: int - the side length of the board
        gamma: float - the discount rate applied to rewards from the neighbouring state
        alpha: float - the step size to update the weights
        epsilon: float - the probability of exploring (rather than exploiting the Greedy policy)
        epochs: int - The number of completed games to simulate during training

        Return:
        Nothing

        Time Complexity: O(Epochs * N^3)
          as it simulates epochs games of up to N moves, calling FindUniqueEdges which is O(N^2)
        """
        # Validate arguments
        if numPlayers < 1:
            raise RuntimeError(f"There must be at least one player")
        if self.numPlayers == None:
            self.numPlayers = numPlayers
        else:
            if self.numPlayers != numPlayers:
                raise("Number of players can not change between calls to fit")
            
        if boardSize < 3:
            raise RuntimeError(f"The minimum board size is 3x3")
        if self.boardSize == None:
            self.boardSize = boardSize
        else:
            if self.boardSize != boardSize:
                raise("Board size can not change between calls to fit")
        
        if epochs < 0:
            raise RuntimeError("The number or epochs must be positive")        
        self.epochs += epochs

        if alpha <= 0.0 or alpha > 1.0:
            raise RuntimeError("Alpha should be in range (0, 1]")

        if gamma < 0.0 or gamma > 1.0:
            raise RuntimeError("Gamma should be in range [0, 1]")

        if epsilon <= 0.0 or epsilon > 1.0:
            raise RuntimeError("Epsilon should be small and in range (0, 1]")

        for _ in range(epochs):
            board = [0] * (boardSize * boardSize)
            board, boardHash = hash(board, boardSize, numPlayers)
            currentPlayer = 1

            while not gameOver(board, boardSize):
                # Selecta a move with epsilon-Greedy strategy
                _, _, indices = findUniqueEdges(board, boardSize, currentPlayer, numPlayers)
                if boardHash in self.policy.keys():
                    currentPolicy = self.policy[boardHash]
                    if random.random() < epsilon:
                        index = random.randrange(len(indices))
                    else:
                        index = np.random.choice(np.flatnonzero(currentPolicy == np.max(currentPolicy)))
                else:
                    currentPolicy = createZeroPolicy(len(indices))
                    self.policy[boardHash] = currentPolicy
                    index = random.randrange(len(indices))
                action = indices[index]
            
                # Process the selected move
                newBoard = board.copy()
                newBoard[action] = currentPlayer
                newBoard, newHash = hash(newBoard, boardSize, numPlayers)
                currentPlayer = (currentPlayer % numPlayers) + 1

                # Take turns for the environment (other players)
                while currentPlayer != 1 and not gameOver(newBoard, boardSize):
                    newAction = random.randrange(len(findEmptyCells(newBoard, boardSize)))
                    newBoard[newAction] = currentPlayer
                    newBoard, newHash = hash(newBoard, boardSize, numPlayers)
                    currentPlayer = (currentPlayer % numPlayers) + 1

                # Find max_a Q(S', a)
                if gameOver(newBoard, boardSize):
                    win = winner(newBoard, boardSize)
                    maxQSa = 1.0 if win == 1 else -1.0
                else:
                    maxQSa = None
                    _, _, actions = findUniqueEdges(newBoard, boardSize, currentPlayer, numPlayers)
                    for a in actions:
                        boardA = newBoard.copy()
                        boardA[a] = currentPlayer
                        boardA, aHash = hash(boardA, boardSize, numPlayers)
                        if gameOver(boardA, boardSize):
                            win = winner(boardA, boardSize)
                            reward = 1.0 if win == 1 else -1.0
                            if maxQSa is None or reward > maxQSa:
                                maxQSa = reward
                        else:
                            # Here we assume only two players
                            otherPlayer = (currentPlayer % numPlayers)+1
                            _, _, otherActions = findUniqueEdges(boardA, boardSize, otherPlayer, numPlayers)
                            for b in otherActions:
                                boardB = boardA.copy()
                                boardB[b] = otherPlayer
                                boardB, bHash = hash(boardB, boardSize, numPlayers)
                                reward = 0.0
                                if gameOver(boardA, boardSize):
                                    win = winner(boardA, boardSize)
                                    reward = 1.0 if win == 1 else -1.0
                                if maxQSa is None or reward > maxQSa:
                                    maxQSa = reward

                # Update policy
                # Reward is +1 for a win, -1 for a loss, 0 for a draw or incomplete game
                if gameOver(newBoard, boardSize):
                    win = winner(newBoard, boardSize)
                    reward = 1.0 if win == 1 else -1.0
                else:
                    reward = 0.0
                # Q learning: Q(S, A) = Q(S, A) + alpha(R + gamma*max_a_Q(S', a) - Q(S, A))
                update = alpha * (reward + gamma * maxQSa - currentPolicy[index])
                self.policy[boardHash][index] += update

                # Prepare for next iteration
                board = newBoard
                boardHash = newHash

        
    def play(self):
        # Allow the user to play a game against the trained model
        # Time Complexity: O(N^3)
        #  as it simulates one game of up to N moves, calling FindUniqueEdges which is O(N^2)
        #  although it is potentially non-terminating if the human player has analysis paralysis
        numPlayers = self.numPlayers
        boardSize = self.boardSize

        board = [0] * (boardSize * boardSize)
        boardHash = 0
        board, boardHash = hash(board, boardHash, numPlayers)
        currentPlayer = 1

        while not gameOver(board, boardSize):
            printBoard(board, boardSize)
            action = -1
            index = -1
            if currentPlayer == 2:
                moves = findEmptyCells(board, boardSize)
                while action not in moves:
                    try:
                        action = int(input(f"Your Turn: "))
                    except ValueError:
                        print(f"Invalid move. Choose from {moves}")
                index = moves.index(action)
            elif currentPlayer == 1:
                result, hashes, indices = findUniqueEdges(board, boardSize, currentPlayer, numPlayers)
                if boardHash in self.policy.keys():
                    currentPolicy = self.policy[boardHash]
                else:
                    currentPolicy = createRandomPolicy(len(indices))

                for i in range(len(indices)):
                    print(f"Q(S={boardHash}, a={indices[i]}) = {currentPolicy[i]}")
                index = np.random.choice(np.flatnonzero(currentPolicy == np.max(currentPolicy)))
                action = indices[index]
            else:
                raise RuntimeError(f"Invalid player number {currentPlayer}")

            print(f"Player {currentPlayer} moved in space {action}")
            board[action] = currentPlayer
            board, boardHash = hash(board, boardSize, numPlayers)
            currentPlayer = (currentPlayer % numPlayers) + 1

        w = winner(board, boardSize)
        if w == 0:
            print("It's a draw!")
        else:
            print(f"Player {winner(board, boardSize)} won!")


def main(args = None):
    ## Monte Carlo
    numPlayers = 2
    boardSize = 5
    learning_rate = 0.9
    epochs = 12500
    start_time = time.time()
    model = Model()
    model.fit_montecarlo(numPlayers, boardSize, learning_rate, epochs)
    duration = time.time() - start_time
    print(f"Training Monte Carlo RL took {duration} seconds")
    model.save("montecarlo.model")
    #model.play()


    ## SARSA
    numPlayers = 2
    boardSize = 5
    learning_rate = 0.9
    gamma = 0.5
    epsilon = 0.1
    epochs = 1250
    start_time = time.time()
    model = Model()
    model.fit_qlearning(numPlayers, boardSize, learning_rate, gamma, epsilon, epochs)
    duration = time.time() - start_time
    print(f"Training SARSA took {duration} seconds")
    model.save("sarsa.model")
    #model.play()

    ## Q Learning
    numPlayers = 2
    boardSize = 5
    learning_rate = 0.9
    gamma = 0.5
    epsilon = 0.1
    epochs = 1250
    start_time = time.time()
    model = Model()
    model.fit_qlearning(numPlayers, boardSize, learning_rate, gamma, epsilon, epochs)
    duration = time.time() - start_time
    print(f"Training Q Learning took {duration} seconds")
    model.save("qlearning.model")
    #model.play()

  

if __name__=="__main__":
    main()


### tictactoe.py
import digraph

# Calculate the hash for a particular board
# hash = sum(board[i] * (num_players+1)^i), where i in [0, board_size^2] and cell values in [0, num_players+1]
# Time complexity: Theta(N)
def getHash(board, board_size, num_players):
    total = 0
    for i in range(board_size * board_size):
        total = total + board[i] * pow(num_players+1, i)
    return total


# Return the board flipped along the centre vertical column
# Time Complexity: Theta(N)
def mirrorBoard(board, board_size):
    result = [0] * len(board)
    for x in range(board_size):
        for y in range(board_size):
            result[(board_size-1-x) + (board_size * y)] = board[x + board_size * y]
    return result


# Return the board rotated clockwise by 90 degrees
# Time Complexity: Theta(N)
def rotateBoardClockwise(board, board_size):
    # Create the output array
    result = [0] * (board_size * board_size)
    
    # Place the centre cell in the new board for odd sized boards
    if board_size % 2 == 1:
        location = board_size//2
        result[location + location * board_size] = board[location + location * board_size]
    
    # Now start along the main diagonal from the top left but not including the centre
    for i in range(board_size//2):
        # Work around the square moving pieces into their new positions
        for j in range(board_size-1-2*i):
            location1 = (i+j) + (i * board_size)
            location2 = (board_size - 1 - i) + ((i+j) * board_size)
            location3 = (board_size - 1 - i - j) + ((board_size - 1 - i) * board_size)
            location4 = (i) + ((board_size - 1 - i - j) * board_size)

            result[location1] = board[location4]
            result[location2] = board[location1]
            result[location3] = board[location2]
            result[location4] = board[location3]

    return result

# Compare the 8 equivalent arrangements of boards
# Return the arrangement with the lowest hash and its hash value
# This reduces the number of states considering symmetry in horizontal, vertical, both diagonals and in combination with mirroring
# Time Complexity: Theta(N)
def hash(board, board_size, num_players):
    best_board = board
    best_hash = getHash(board, board_size, num_players)
    mirror = mirrorBoard(board, board_size)
    mirror_hash = getHash(mirror, board_size, num_players)
    if mirror_hash < best_hash:
        best_board = mirror
        best_hash = mirror_hash
    for i in range(3):
        board = rotateBoardClockwise(board, board_size)
        board_hash = getHash(board, board_size, num_players)
        if board_hash < best_hash:
            best_board = board
            best_hash = board_hash
        mirror = rotateBoardClockwise(mirror, board_size)
        mirror_hash = getHash(mirror, board_size, num_players)
        if mirror_hash < best_hash:
            best_board = mirror
            best_hash = mirror_hash
    return best_board, best_hash


# Return True if the game is over, otherwise False
# Time Complexity: O(N)
def gameOver(board, board_size):
    # Horizontal
    for y in range(board_size):
        count = 1
        target = board[y*board_size]
        if target != 0:
            for x in range(1, board_size):
                if board[x + y*board_size] == target:
                    count += 1
                else:
                    break
        if count == board_size:
            return True
            
    # Vertical
    for x in range(board_size):
        count = 1
        target = board[x]
        if target != 0:
            for y in range(1, board_size):
                if board[x + y*board_size] == target:
                    count += 1
                else:
                    break
        if count == board_size:
            return True

    #Diagonals
    count = 1
    target = board[0]
    if target != 0:
        for i in range(1, board_size):
            if board[i + board_size*i] == target:
                count += 1
            else:
                break
    if count == board_size:
        return True
    
    count = 1
    target = board[board_size - 1]
    if target != 0:
        for i in range(1, board_size):
            if board[board_size - 1 - i + board_size*i] == target:
                count += 1
            else:
                break
    if count == board_size:
        return True
    
    # Nobody has won, the game is not over if there is an empty space
    for i in range(board_size * board_size):
        if board[i] == 0:
            return False
        
    # The board is full, so the game is over
    return True


# Returns a list of indices of all empty cells in the board
# Time Complexity: O(N)
def findEmptyCells(board, board_size):
    result = []
    for i in range(board_size * board_size):
        if board[i] == 0:
            result.append(i)
    return result


# Return the new board state after the specified player makes a given move
# Time complexity: Theta(1)
def move(board, board_size, index, player, num_players):
    if index < 0 or index > board_size * board_size or board[index] != 0 or player < 1 or player > num_players:
        raise Exception("Preconditions violated")
    board[index] = player
    return board


# Return True if the value is in the list, otherwise False
# Linear search is used rather than binary search of a sorted list, or a hash table as the list is short
# Time Complexity: O(K) (where K is number of keys)
def find(keys, value):
    for e in keys:
        if e == value:
            return True
    return False


# Return the player who won [1 to numPlayers], or 0 for a draw
# Should not be called when the game is still in play (throws exception)
# Time complexity: O(N)
def winner(board, board_size):
    # Horizontal
    for y in range(board_size):
        count = 1
        target = board[y*board_size]
        if target != 0:
            for x in range(1, board_size):
                if board[x + y*board_size] == target:
                    count += 1
                else:
                    break
        if count == board_size:
            return target
            
    # Vertical
    for x in range(board_size):
        count = 1
        target = board[x]
        if target != 0:
            for y in range(1, board_size):
                if board[x + y*board_size] == target:
                    count += 1
                else:
                    break
        if count == board_size:
            return target

    #Diagonals
    count = 1
    target = board[0]
    if target != 0:
        for i in range(1, board_size):
            if board[i + board_size*i] == target:
                count += 1
            else:
                break
    if count == board_size:
        return target
    
    count = 1
    target = board[board_size - 1]
    if target != 0:
        for i in range(1, board_size):
            if board[board_size - 1 - i + board_size*i] == target:
                count += 1
            else:
                break
    if count == board_size:
        return target
    
    # Nobody has won, the game is not over if there is an empty space
    # Throw an exception as this means the precondition was violated
    for i in range(board_size * board_size):
        if board[i] == 0:
            raise RuntimeError("Preconditions violated")
        
    # The board is full, so the game is over
    return 0


# Returns a list of all possible subsequent states and their hash for the given player, and the index on the original board
#  Excludes states which are identical based on symmetry
# Returns no moves if the game is over
# Time Complexity: O(N^2)
def findUniqueEdges(board, board_size, player, num_players):
    result = []
    hashes = []
    indices = []
    if gameOver(board, board_size):
        return result, hashes, indices		# If the board is full, or game is won there are no moves left
    available = findEmptyCells(board, board_size)
    
    for i in available:
        clone = board.copy()
        new_board, board_hash = hash(move(clone, board_size, i, player, num_players), board_size, num_players)
        if not find(hashes, board_hash):
            hashes.append(board_hash)
            result.append(new_board)
            indices.append(i)
    return result, hashes, indices


# Return the utility of the result to the current player
# Time Complexity; O(1)
def score(winner, player):
    if winner == player:
        return 1
    elif winner == 0:
        return 0
    else:
        return -1


# Evaluate the game tree from the given state and player and return its score, selected move and cell index
# Recursive maximin algorithm
# Rewards for leaf nodes: 1 = win for AI, -1 = loss for AI, 0 = draw
# If no moves are available, the returned board is unchanged and cell index is None
def evaluate(board, board_hash, board_size, player, target_player, num_players):
    digraph.setNodeLabel(board_hash, board)

    if gameOver(board, board_size):
        board_score = score(winner(board, board_size), target_player)
        digraph.setNodeColour(board_hash, board_score)
        return score(winner(board, board_size), target_player), board, None
    
    edges, hashes, indices = findUniqueEdges(board, board_size, player, num_players)

    max_score = -2
    max_board = None
    max_index = None
    min_score = 2
    min_board = None
    min_index = None

    for i in range(len(edges)):
        digraph.setNodeEdge(board_hash, hashes[i])
        nextPlayer = (player % num_players) + 1
        board_score, new_board, new_index = evaluate(edges[i], hashes[i], board_size, nextPlayer, target_player, num_players)
        
        if board_score < min_score:
            min_score = board_score
            min_board = edges[i]
            min_index = indices[i]
        if board_score > max_score:
            max_score = board_score
            max_board = edges[i]
            max_index = indices[i]

    if player == target_player: # AI player - return the maximum child
        digraph.setNodeColour(board_hash, max_score);
        return max_score, max_board, max_index
    else:		# Other player - return minimum child
        digraph.setNodeColour(board_hash, min_score);
        return min_score, min_board, min_index

### Graph.py

from typing import Self

class Edge():
    """
    An edge in the Graph
    """

    def __init__(self, u: "Node", v: "Node", w: float, reverse = None):
        """
        The edge connects vertices u to v with weight w
        If the edge is part of a bidirectional pair, reversed will point to the antiparallel edge
        """
        self.u = u
        self.v = v
        self.w = w
        self.reverse = reverse

    def __str__(self):
        """ Return a string representation of the vertex """        
        return str(f"({self.u}->{self.v} w:{self.w})")


class Node():
    """ A vertex in the Graph """

    def __init__(self, label: str):
        """
        Each vertex is labelled with a unique string (for printing) and begins without any edges
        """
        self.value = label
        self.edges = []
        self.weight = 0.0

    def findEdge(self, u: Self, v: Self):
        """
        Find an edge given the source and destinatino nodes

        Time Complexity: O(degree(U))
        """
        for e in self.edges:
            if e.u.value == u.value and e.v.value == v.value:
                return e
        return None
    

    def findEdgeByValue(self, v: str):
        """
        Find an edge given the value of the destination

        Time Complexity: O(degree(U))
        """
        for e in self.edges:
            if e.u.value == self.value and e.v.value == v:
                return e
        return None


    def __str__(self):
        """ Return a string representation (label) of the vertex """        
        return str(self.value)


class Graph():
    """ A graph G = (V, E)"""


    def __init__(self):
        """ Initialise an empty Graph"""
        self.vertices = []
        self.map = {}


    def addVertex(self, v: "Node"):
        """ Add vertex V to the Graph """
        self.vertices.append(v)
        self.map[v.value] = self.vertices[-1]


    def addBiEdge(self, u: "Node", v: "Node", w: float):
        """
        Add a bidirectional edge to the graph

        The edge connects u to v and v to u with weight w
        Internally this is implemented using two edges, each of which holds a pointer to the other
        """
        a = self.addDiEdge(u, v, w)
        b = self.addDiEdge(v, u, w, a)
        a.reverse = b


    def addDiEdge(self, u: "Node", v: "Node", w: float, reverse = None) -> "Edge":
        """
        Add a directed edge to the graph
        
        The edge connects u to v with weight w.
        It creates an outgoing edge, so u->v will be stored in u, and v->u will be stored in vs
        If the edge is part of a bidirectional pair, reversed will point to the antiparallel edge

        Returns the created edge
        """
        edge = Edge(u, v, w, reverse)
        u.edges.append(edge)
        u.weight += w
        return edge


    def __str__(self):
        """ String representation of a Graph """
        total = 0.0
        edgeTotal = 0.0
        ret = "Graph with:"
        ret += "\nVertices: [ "
        for v in self.vertices:
            ret += str(f"{v} w:{v.weight} ")
        ret += "]\nEdges:\t [ "
        for v in self.vertices:
            total += v.weight
            for e in v.edges:
                edgeTotal += e.w
                ret += str(e) + " "                
        ret += "]\n"
        ret += str(f"Total v:{total} e:{edgeTotal}\n")
        return ret

### Sparse_graph.py

#!/usr/bin/env python3

from collections import deque
import os
import sys
from typing import List

# Since I am using non standard directories for grouping classes, add them to path
from os.path import dirname
sys.path.append(os.path.join(dirname(__file__), r'../Cut'))
from graph import Graph, Node, Edge
from tictactoe import hash, findUniqueEdges, findEmptyCells

def main(args = None):
    numPlayers = 2
    boardSize = 5

    # Set up the starting, empty board
    board = [0] * (boardSize * boardSize)
    board, boardHash = hash(board, boardSize, numPlayers)
    current = Node(boardHash)
    current.board = board
    current.player = 1
    level = -1

    graph = Graph()
    graph.addVertex(current)

    queue = deque()
    queue.append(current)

    while len(queue) > 0:
        current = queue.popleft()

        currentLevel = boardSize * boardSize - len(findEmptyCells(current.board, boardSize))
        if currentLevel > level:
            level = currentLevel
            print(f"Level {currentLevel} States {len(queue)+1}")

        newPlayer = (current.player % numPlayers) + 1 
        result, hashes, indices = findUniqueEdges(current.board, boardSize, current.player, numPlayers)
        for i in range(len(indices)):
            if hashes[i] in graph.map:
                newNode = graph.map[hashes[i]]
                assert newNode.player == newPlayer
            else:
                newNode = Node(hashes[i])
                newNode.player = newPlayer
                newNode.board = result[i]
                graph.addVertex(newNode)
                queue.append(newNode)

            newEdge = Edge(current, newNode, 0.0)
            newEdge.action = indices[i]
            current.edges.append(newEdge)

    print(f"Number of states: {len(graph.vertices)}")
    #print(graph)

if __name__=="__main__":
    main()
