"""
This file contains all data generation methods for the assignment
"""

import random
import math

def generate_random():
    return random.random()

# numpy multivariate normal
def box_mueller_number(m, s):
    random_number = generate_random()
    x1 = 2.0 * random_number - 1.0
    x2 = 2.0 * random_number - 1.0
    w = x1 ** 2 + x2 ** 2
    if w >= 1.0:
        w = math.sqrt( (2.0 * math.log(w)) / w)
        y1 = x1 * w
        y2 = x2 * w
        return m + y1 * s
    return None

def get_box_mueller(n, m, s):
    numbers = []

    while len(numbers) < n:
        val = box_mueller_number(m, s)
        if val != None:
            numbers.append(val)
    return numbers
