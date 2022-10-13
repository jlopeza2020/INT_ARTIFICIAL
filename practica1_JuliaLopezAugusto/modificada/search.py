# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't 
    implement any of the methods (in object-oriented terminology: an abstract 
    class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", 
            problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    # use a stack to store all states due to the algorithm we are 
    # looking for has the shape of an stack (Last In First Out)
    frontier = util.Stack()

    # array that stores all explored nodes
    expanded = []

    # set start position
    start_state = problem.getStartState()
    # set an array for storing pacman actions
    pcm_actions = []

    # I create a node in which will be stored state and actions 
    # that the pacman will perform
    start_node = (start_state, pcm_actions)
    
    # I insert them into the stack
    frontier.push(start_node)

    # while there are positions in stack 
    while not frontier.isEmpty():
        # it is being explored the last node being pushed 
        # (as it is an stack) 
        node = frontier.pop()

        current_state = node[0]
        actions = node [1]

        # if current state has not being explored yet
        if current_state not in expanded:
            # set current node as explored
            expanded.append(current_state)

            # if it has reached to the goal (if is True), returns actions
            # for the pacman to be performed
            if problem.isGoalState(current_state):
                break
            else:
                # get list of successor nodes in 
                # form (successor, action, stepCost)
                successors = problem.getSuccessors(current_state)
                
                # push each successor (state, action) to frontier
                for sucsr_state, sucsr_action, sucsr_cost in successors:
                    new_action = actions + [sucsr_action]
                    new_node = (sucsr_state, new_action)

                    frontier.push(new_node)

    return actions  

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    # use a queue to store all states due to the algorithm we are 
    # looking for has the shape of an queue (First In First Out)
    frontier = util.Queue()

    # array that stores all explored nodes
    expanded = []

    # set start position
    start_state = problem.getStartState()
    # set an array for storing  pacman actions
    pcm_actions = []

    # I create a node in which will be stored state and actions 
    # that the pacman will perform
    start_node = (start_state, pcm_actions)

    # I insert them into the queue
    frontier.push(start_node)

    # while there are positions in queue
    while not frontier.isEmpty():
        # it is being exploring the first node being pushed 
        # (as it is a queue) 
        node = frontier.pop()
        
        current_state = node[0]
        actions = node[1]
        
        # if current state has not being explored yet
        if current_state not in expanded:
            # set current node as explored
            expanded.append(current_state)

            # if it has reached to the goal, returns actions
            # for the pacman to be performed
            if problem.isGoalState(current_state):
                break
            else:
                # get list of successor nodes in 
                # form (successor, action, stepCost)
                successors = problem.getSuccessors(current_state)
                
                # push each successor (state, action, cost) to frontier
                for sucsr_state, sucsr_action, sucsr_cost in successors:
                    new_action = actions + [sucsr_action]
                    new_node = (sucsr_state, new_action)

                    frontier.push(new_node)

    return actions  

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    # use a Queue(LIFO) with priority to store all states since we need an 
    # algorithm that should store cost to got firstly to the ones with lower
    # cost
    frontier = util.PriorityQueue()

    # array that stores all explored nodes
    expanded = []

    # set start position
    start_state = problem.getStartState()
    # set an array for storing pacman actions
    pcm_actions = []

    #set initical Cost
    initial_cost = 0

    # I create a node in which will be stored state, 
    # actions and cost that the pacman will perform
    start_node = (start_state, pcm_actions, initial_cost)

    # I insert them into the queue
    # def push(self, item, priority)
    frontier.push(start_node, initial_cost)

    # while there are positions in queue
    while not frontier.isEmpty():
        # it is being exploring the first node being pushed 
        # (as it is a queue) 
        node = frontier.pop()
        
        current_state = node[0]
        actions = node[1]

        # if current state has not being explored yet 
        if current_state not in expanded:
            # set current node as explored
            expanded.append(current_state) 

            # if it has reached to the goal, returns actions
            # for the pacman to be performed
            if problem.isGoalState(current_state):
                break
            else:
                # get list of successor nodes in 
                # form of (successor, action, stepCost)
                successors = problem.getSuccessors(current_state)
                
                # push each successor (state, action, cost) to frontier
                for sucsr_state, sucsr_action, sucsr_cost in successors:
                    new_action = actions + [sucsr_action]
                    total_cost = problem.getCostOfActions(new_action)
                    new_node = (sucsr_state, new_action, total_cost)

                    # better than push since it has implemented push
                    # and others terms
                    frontier.update(new_node, total_cost)

    return actions

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the 
    nearest goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    # use a Queue(LIFO) with priority to store all states
    frontier = util.PriorityQueue()

    # array that stores all explored nodes
    expanded = []

    # set start position
    start_state = problem.getStartState()
    # set an array for storing pacman actions
    pcm_actions = []

    # set initical cost composed by initial cost = 0 + heuristic
    # heuristic is a function given as a parameter 
    # composed by start state andthe problem itself
    initial_cost = 0 + heuristic(start_state, problem)

    # I create a node in which will be stored 
    # state and actions that the pacman will perform
    start_node = (start_state, pcm_actions, initial_cost)

    # def push(self, item, priority)
    frontier.push(start_node, initial_cost)

    # while there are positions in queue
    while not frontier.isEmpty():
        # it is being exploring the first node being pushed 
        # (as it is a queue) 
        node = frontier.pop()
        
        current_state = node[0]
        actions = node[1]
 
        # if current state has not being explored yet 
        if current_state not in expanded:
            # set current node as explored
            expanded.append(current_state)

            # if it has reached to the goal, returns actions
            # for the pacman to be performed
            if problem.isGoalState(current_state):
                break
            else:
                # get list of successor nodes in 
                # form of (successor, action, stepCost)
                successors = problem.getSuccessors(current_state)
                
                # push each successor (state, action, cost) to frontier
                for sucsr_state, sucsr_action, sucsr_cost in successors:
                    new_action = actions + [sucsr_action]

                    # getCostofActions returns the total cost of 
                    # a particular sequence of actions and we have 
                    # to add to it the value of the heuristic
                    total_cost = problem.getCostOfActions(new_action) + heuristic(sucsr_state, problem)

                    new_node = (sucsr_state, new_action, total_cost)

                    frontier.update(new_node, total_cost)

    return actions

    


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
