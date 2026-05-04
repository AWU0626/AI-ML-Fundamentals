############################################################
# CIS 521: Homework 9
############################################################

import homework9_data as data

student_name = "Aaron Wu"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.

############################################################
# Section 1: Perceptrons
############################################################


class BinaryPerceptron(object):

    def __init__(self, examples, iterations):
        self.weights = {}

        for example in examples:
            for feature in example[0]:
                self.weights[feature] = 0

        for i in range(iterations):
            for example in examples:
                x, y = example[0], example[1]
                prediction = self.predict(x)

                if prediction != y:
                    for feature in x:
                        if y:
                            self.weights[feature] += x[feature]
                        else:
                            self.weights[feature] -= x[feature]

    def predict(self, x):
        res = 0

        for feature in x:
            res += self.weights[feature] * x[feature]

        if res > 0:
            return True

        return False


class MulticlassPerceptron(object):

    def __init__(self, examples, iterations):
        self.weights = {}

        for x, y in examples:
            if y not in self.weights:
                self.weights[y] = {}
                for feature in x:
                    self.weights[y][feature] = 0

        for i in range(iterations):
            for x, y in examples:
                prediction = self.predict(x)

                if prediction != y:
                    for feature in x:
                        self.weights[y][feature] += x[feature]
                        self.weights[prediction][feature] -= x[feature]

    def predict(self, x):
        max_val, max_label = float('-inf'), None

        for label in self.weights:
            res = 0

            for feature in x:
                if feature not in self.weights[label]:
                    self.weights[label][feature] = 0
                res += self.weights[label][feature] * x[feature]

            if res > max_val:
                max_val, max_label = res, label

        return max_label

############################################################
# Section 2: Applications
############################################################


class IrisClassifier(object):

    def __init__(self, data):
        training_data = []

        for x, y in data:
            hashmap = {}

            for i in range(len(x)):
                hashmap[i] = x[i]

            training_data.append((hashmap, y))

        self.perceptron = MulticlassPerceptron(training_data, 40)

    def classify(self, instance):
        hashmap = {}

        for i in range(len(instance)):
            hashmap[i] = instance[i]

        return self.perceptron.predict(hashmap)


class DigitClassifier(object):

    def __init__(self, data):
        training_data = []

        for x, y in data:
            hashmap = {}

            for i in range(len(x)):
                hashmap[i] = x[i]

            training_data.append((hashmap, y))

        self.perceptron = MulticlassPerceptron(training_data, 15)

    def classify(self, instance):
        hashmap = {}

        for i in range(len(instance)):
            hashmap[i] = instance[i]

        return self.perceptron.predict(hashmap)


class BiasClassifier(object):

    def __init__(self, data):
        training_data = []

        for x, y in data:
            hashmap = {0: x - 1}
            training_data.append((hashmap, y))

        self.perceptron = BinaryPerceptron(training_data, 10)

    def classify(self, instance):
        hashmap = {0: instance - 1}

        return self.perceptron.predict(hashmap)


class MysteryClassifier1(object):

    def __init__(self, data):
        training_data = []

        for x, y in data:
            hashmap = {0: x[0] * x[0] + x[1] * x[1] - 4}
            training_data.append((hashmap, y))

        self.perceptron = BinaryPerceptron(training_data, 10)

    def classify(self, instance):
        init = instance[0] * instance[0] + instance[1] * instance[1]
        hashmap = {0: init - 4}

        return self.perceptron.predict(hashmap)


class MysteryClassifier2(object):

    def __init__(self, data):
        training_data = []

        for x, y in data:
            hashmap = {0: x[0] * x[1] * x[2]}
            training_data.append((hashmap, y))

        self.perceptron = BinaryPerceptron(training_data, 10)

    def classify(self, instance):
        hashmap = {0: instance[0] * instance[1] * instance[2]}
        return self.perceptron.predict(hashmap)

############################################################
# Section 3: Feedback
############################################################


# Just an approximation is fine.
feedback_question_1 = """
4
"""

feedback_question_2 = """
This one is fine. The pseudo code is already given.
"""

feedback_question_3 = """
The entirety of section 2, I see more applications than ever.
"""
