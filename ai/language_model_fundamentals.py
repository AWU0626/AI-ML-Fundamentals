############################################################
# CIS 521: Homework 8
############################################################
import string
import random
import math
from string import whitespace, punctuation

student_name = "Aaron Wu"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.

############################################################
# Section 1: Ngram Models
############################################################


def tokenize(text):
    tokens = []
    current = ""
    for c in text:
        if c in whitespace or c in punctuation:
            if current:
                tokens.append(current)
                current = ""
            if c in punctuation:
                tokens.append(c)
        else:
            current += c

    if current:
        tokens.append(current)

    return tokens


def ngrams(n, tokens):
    lst = []
    for i in range(len(tokens)):
        if i < n - 1:
            prefix = ('<START>',) * (n - 1 - i) + tuple(tokens[:i])
            lst.append((prefix, tokens[i]))
        else:
            lst.append((tuple(tokens[i - n + 1:i]), tokens[i]))
    if n > 1:
        lst.append((tuple(tokens[-n + 1:]), '<END>'))
    if n == 1:
        lst.append(((), '<END>'))
    return lst


class NgramModel(object):

    def __init__(self, n):
        self.n = n
        self.counts = {}

    def update(self, sentence):
        tokens = tokenize(sentence)
        ngrams_lst = ngrams(self.n, tokens)

        for context, token in ngrams_lst:
            if context not in self.counts:
                self.counts[context] = {}
            if token not in self.counts[context]:
                self.counts[context][token] = 1
            else:
                self.counts[context][token] += 1

    def prob(self, context, token):
        if context not in self.counts:
            return 0
        if token not in self.counts[context]:
            return 0

        return (self.counts[context][token] /
                sum(self.counts[context].values()))

    def random_token(self, context):
        if context not in self.counts:
            return None

        tokens = sorted(self.counts[context].keys())
        sum_tokens = sum(self.counts[context].values())
        rand = random.random()
        cumulative_prob = 0

        for i in range(len(tokens)):
            token = tokens[i]
            cumulative_prob += (self.counts[context][token] /
                                sum_tokens)
            if cumulative_prob > rand:
                return tokens[i]

    def random_text(self, token_count):
        start_context = tuple(["<START>"] * (self.n - 1))
        context = start_context
        tokens = []

        for _ in range(token_count):
            token = self.random_token(context)
            tokens.append(token)
            if token == "<END>":
                context = start_context
            elif self.n > 1:
                context = context[1:] + (token,)

        return " ".join(tokens)

    def perplexity(self, sentence):
        tokens = tokenize(sentence)
        ngrams_lst = ngrams(self.n, tokens)
        log_prob = 0

        for context, token in ngrams_lst:
            log_prob += math.log(self.prob(context, token))
        return math.exp(-log_prob / len(ngrams_lst))


def create_ngram_model(n, path):
    ngram_model = NgramModel(n)
    with open(path, 'r') as file:
        for line in file:
            ngram_model.update(line)

    return ngram_model

############################################################
# Section 2: Feedback
############################################################


# Just an approximation is fine.
feedback_question_1 = """
7
"""

feedback_question_2 = """
getting the ngram process & its probabilities behind the scene
"""

feedback_question_3 = """
The math for ngram probability
"""
