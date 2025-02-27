#!/usr/bin/env python3
import math


def roundup(x):
    return math.ceil(x / 10.0) * 10


def rounddown(x):
    return math.floor(x / 10.0) * 10
