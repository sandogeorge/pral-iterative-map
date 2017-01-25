"""Implements miscellaneous helper functions"""


import math


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def checklineIntersection(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def lineIntersection(l1, l2):
    xdiff = (l1[0][0] - l1[1][0], l2[0][0] - l2[1][0])
    ydiff = (l1[0][1] - l1[1][1], l2[0][1] - l2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)

    d = (det(*l1), det(*l2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return (x, y)


def rotatePoint(point, origin, theta):
    return (
        math.cos(theta) * (point[0] - origin[0]) -
        math.sin(theta) * (point[1] - origin[1]) + origin[0],
        math.sin(theta) * (point[0] - origin[0]) +
        math.cos(theta) * (point[1] - origin[1]) + origin[1]
    )
