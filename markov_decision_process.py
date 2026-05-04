# Include your imports here, if any are used.
import copy

student_name = "Aaron Wu"


# 1. Value Iteration
class ValueIterationAgent:
    """Implement Value Iteration Agent using Bellman Equations."""

    def __init__(self, game, discount):
        """Store game object and discount value into the agent object,
        initialize values if needed.
        """
        self.game = game
        self.discount = discount
        self.values = {}

        for state in game.states:
            self.values[state] = 0

    def get_value(self, state):
        """Return value V*(s) correspond to state.
        State values should be stored directly for quick retrieval.
        """
        return self.values[state]

    def get_q_value(self, state, action):
        """Return Q*(s,a) correspond to state and action.
        Q-state values should be computed using Bellman equation:
        Q*(s,a) = Σ_s' T(s,a,s') [R(s,a,s') + γ V*(s')]
        """
        transition_probabilities = self.game.get_transitions(state, action)
        q = 0

        for t, prob in transition_probabilities.items():
            if t in self.values:
                q += prob * (self.game.get_reward(state, action, t)
                             + self.discount * self.get_value(t))
            else:
                q += prob * (self.game.get_reward(state, action, t))

        return q

    def get_best_policy(self, state):
        """Return policy π*(s) correspond to state.
        Policy should be extracted from Q-state values using policy extraction:
        π*(s) = argmax_a Q*(s,a)
        """
        if state not in self.values:
            return None
        else:
            values = {}
            actions = self.game.get_actions(state)

            for action in actions:
                values[action] = self.get_q_value(state, action)

            return max(values, key=values.get)

    def iterate(self):
        """Run single value iteration using Bellman equation:
        V_{k+1}(s) = max_a Q*(s,a)
        Then update values: V*(s) = V_{k+1}(s)
        """
        new_values = copy.deepcopy(self.values)

        for state in self.values:
            best_policy = self.get_best_policy(state)
            new_values[state] = self.get_q_value(state, best_policy)

        self.values = new_values


# 2. Policy Iteration
class PolicyIterationAgent(ValueIterationAgent):
    """Implement Policy Iteration Agent.

    The only difference between policy iteration and value iteration is at
    their iteration method. However, if you need to implement helper function
    or override ValueIterationAgent's methods, you can add them as well.
    """

    def iterate(self):
        """Run single policy iteration.
        Fix current policy, iterate state values V(s) until
        |V_{k+1}(s) - V_k(s)| < ε
        """
        epsilon = 1e-6

        while True:
            new_values = copy.deepcopy(self.values)

            for state in self.values:
                best_policy = self.get_best_policy(state)
                new_values[state] = self.get_q_value(state, best_policy)

            iterate = False
            for state in self.values:
                abs_value = abs(new_values.get(state) - self.values.get(state))
                if abs_value > epsilon:
                    iterate = True
                    break

            self.values = new_values

            if not iterate:
                break


# 3. Bridge Crossing Analysis
def question_3():
    discount = 0.9
    noise = 0.015
    return discount, noise


# 4. Policies
def question_4a():
    discount = 0.2
    noise = 0.0
    living_reward = 0.0
    return discount, noise, living_reward
    # If not possible, return 'NOT POSSIBLE'


def question_4b():
    discount = 0.2
    noise = 0.2
    living_reward = 0.2
    return discount, noise, living_reward
    # If not possible, return 'NOT POSSIBLE'


def question_4c():
    discount = 0.9
    noise = 0.0
    living_reward = 0.0
    return discount, noise, living_reward
    # If not possible, return 'NOT POSSIBLE'


def question_4d():
    discount = 0.2
    noise = 0.2
    living_reward = 0.7
    return discount, noise, living_reward
    # If not possible, return 'NOT POSSIBLE'


def question_4e():
    discount = 0.9
    noise = 0.0
    living_reward = 2
    return discount, noise, living_reward
    # If not possible, return 'NOT POSSIBLE'


# 5. Feedback
# Just an approximation is fine.
feedback_question_1 = """
6 hours
"""

feedback_question_2 = """
configuring tkinter for python came as a bit of roadblock.
"""

feedback_question_3 = """
I like the stub giving us the direction. also the equations
so it's easier to write out the process. Trying around the
parameter's also pretty fun.
"""
