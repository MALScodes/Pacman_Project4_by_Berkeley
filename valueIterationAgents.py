# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations): # Iterate for the specified number of times
            new_values = self.values.copy() # Create a new dictionary with the same values as the current value function estimates
            for state in self.mdp.getStates(): # For each state in the MDP
                if self.mdp.isTerminal(state): # If the state is a terminal state, set its value estimate to 0
                    new_values[state] = 0 # Otherwise, compute the maximum value estimate over all possible actions in the state
                else:
                    arrowvalue = []
                    for action in self.mdp.getPossibleActions(state):
                        arrowvalue.append(self.getQValue(state, action))
                    new_values[state] = max(arrowvalue)
            self.values = new_values # Update the value function estimates with the newly computed values


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        possible_next_states = self.mdp.getTransitionStatesAndProbs(state, action) # Get the possible next states and their transition probabilities
        value_estimates = [] # Create an empty list to store the value estimates for each possible next state
        for next_state, T in possible_next_states: # For each possible next state and its probability of transitioning from the current state with the given action
            discounted_future_value = self.discount * self.values[next_state] # Compute the discounted future value of the next state using the current value function estimates
            reward = self.mdp.getReward(state, action, next_state) # Compute the reward obtained by taking the given action in the current state and transitioning to the next state
            value_estimate = T * (reward + discounted_future_value) # Compute the value estimate for the next state as the sum of the discounted future value and the immediate reward
            value_estimates.append(value_estimate) # Add the value estimate to the list of value estimates for all possible next states
        return sum(value_estimates) # Return the sum of all value estimates for all possible next states

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        possible_actions = self.mdp.getPossibleActions(state) # Get all possible actions for the given state

        if not possible_actions: # If there are no possible actions for the given state, return None
            return None

        # Compute the Q-value for each possible action for the given state
        # using the self.getQValue() method, and store these in a list of tuples
        arrowvalue = [(action, self.getQValue(state, action)) for action in possible_actions]

        # Find the action with the highest Q-value among the possible actions
        # using the max() function and a lambda function to access the second element of each tuple
        # and return the corresponding action (i.e., the first element of the tuple)
        max_value = float('-inf')
        max_element = None
        for element in arrowvalue:
            if element[1] > max_value:
                max_value = element[1]
                max_element = element
        result = max_element[0]
        return result # Return the action with the highest Q-value for the given state

        
    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # mdp.getStates()
        list_of_states = self.mdp.getStates()# Obtain the list of states from the MDP object and initialize the value of each state to 0.
        self.values = {state: 0 for state in self.mdp.getStates()}# Initialize the value of each state to 0 using a list comprehension.
        states_i = len(self.values)# Count the number of states in the MDP.
        itterations_i= range(self.iterations)  # Create a range object with the number of iterations to perform.

        for i in itterations_i: # Iterate over the range of iterations.
            firstState= list_of_states[0] # Get the first state 
            # mdp.getPossibleActions(state)
            # mdp.getTransitionStatesAndProbs(state, action)
            currentStateLoc = i % states_i # Calculate the index of the current state based on the iteration number.
            currentState = list_of_states[currentStateLoc] # Get the current state based on the index.
            while 1:
                terminal = self.mdp.isTerminal(currentState) # Determine if the current state is a terminal state.
                break
            def case1(): # Define different cases for updating the value of the current state.
                arrows = self.getAction(currentState) # Get the action with the maximum Q-value for the current state.
                # valueOFq = self.getQValue(currentState, action) # Get the Q-value of the current state and the chosen action.
                self.values[currentState] = self.getQValue(currentState, arrows) # Update the value of the current state to be the Q-value of the chosen action.


            def case2():
                arrows = self.getAction(firstState)
                self.values[firstState] = self.getQValue(firstState, arrows)

            def default(): # Define a default case for situations where the current state is a terminal state.
                pass

            switcher = {
                not terminal: case1,
                # add more cases if necessary
            }

            func = switcher.get(True, default) # Get the function to execute based on the value of "not terminal" in the switcher dictionary.

            # Execute the function
            func()
class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"