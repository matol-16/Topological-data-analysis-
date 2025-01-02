import random
import numpy as np


## QUESTION 1:

class Sphere:
    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = radius

    def contains(self, point):
        """Check if the sphere contains the given point."""
        return np.linalg.norm(self.center - np.array(point)) <= self.radius + 1e-9

def make_sphere_two_points(p1, p2):
    """Create a sphere with two points on its boundary."""
    center = (np.array(p1) + np.array(p2)) / 2
    radius = np.linalg.norm(np.array(p1) - np.array(p2)) / 2
    return Sphere(center, radius)

def make_sphere_three_points(p1, p2, p3):
    """Create a sphere with three points on its boundary."""
    A = np.array(p1)
    B = np.array(p2)
    C = np.array(p3)

    # Solve for the circumcenter of the triangle
    AB = B - A
    AC = C - A
    ABxAC = np.cross(AB, AC)

    if np.linalg.norm(ABxAC) == 0:
        raise ValueError("Points are collinear!")

    circumcenter = (
        np.cross(np.dot(AB, AB) * AC - np.dot(AC, AC) * AB, ABxAC) / (2 * np.linalg.norm(ABxAC)**2)
    )
    center = A + circumcenter
    radius = np.linalg.norm(center - A)
    return Sphere(center, radius)

def make_sphere_four_points(p1, p2, p3, p4):
    """Create a sphere with four points on its boundary."""
    A = np.array(p1)
    B = np.array(p2)
    C = np.array(p3)
    D = np.array(p4)

    # Compute the matrix to solve the circumcenter
    M = np.array([
        [np.dot(A, A), A[0], A[1], A[2], 1],
        [np.dot(B, B), B[0], B[1], B[2], 1],
        [np.dot(C, C), C[0], C[1], C[2], 1],
        [np.dot(D, D), D[0], D[1], D[2], 1]
    ])
    Mx = np.copy(M)
    Mx[:, 1] = 1

    My = np.copy(M)
    My[:, 2] = 1

    Mz = np.copy(M)
    Mz[:, 3] = 1

    Mdet = np.linalg.det(M[:, 1:])
    Mxdet = np.linalg.det(Mx[:, 1:])
    Mydet = np.linalg.det(My[:, 1:])
    Mzdet = np.linalg.det(Mz[:, 1:])

    if np.isclose(Mdet, 0):
        raise ValueError("Points are coplanar or collinear!")

    center = np.array([Mxdet, Mydet, Mzdet]) / (2 * Mdet)
    radius = np.linalg.norm(center - A)
    return Sphere(center, radius)

def trivial(R):
    """Find the minimal sphere for 0, 1, 2, 3, or 4 points."""
    if not R:
        return Sphere([0, 0, 0], 0)
    elif len(R) == 1:
        return Sphere(R[0], 0)
    elif len(R) == 2:
        return make_sphere_two_points(R[0], R[1])
    elif len(R) == 3:
        return make_sphere_three_points(R[0], R[1], R[2])
    elif len(R) == 4:
        return make_sphere_four_points(R[0], R[1], R[2], R[3])
    else:
        raise ValueError("trivial function called with more than 4 points!")

def welzl(P, R):
    """Recursive implementation of Welzl's algorithm for 3D."""
    if not P or len(R) == 4:
        return trivial(R)

    p = P.pop(random.randint(0, len(P) - 1))
    D = welzl(P, R)

    if D.contains(p):
        P.append(p)
        return D

    result = welzl(P, R + [p])
    P.append(p)
    return result

def minimal_enclosing_sphere(points):
    """Compute the minimal enclosing sphere for a set of points."""
    points = points[:]
    random.shuffle(points)
    return welzl(points, [])

# Test cases
def test_minimal_enclosing_sphere():
    """Test cases for minimal enclosing sphere."""
    # Test 1: Single point
    points = [(0, 0, 0)]
    sphere = minimal_enclosing_sphere(points)
    assert np.allclose(sphere.center, [0, 0, 0])
    assert np.isclose(sphere.radius, 0)
    print("Test 1 passed!")

    # Test 2: Two points
    points = [(0, 0, 0), (2, 0, 0)]
    sphere = minimal_enclosing_sphere(points)
    assert np.allclose(sphere.center, [1, 0, 0])
    assert np.isclose(sphere.radius, 1)
    print("Test 2 passed!")

    # Test 3: Three points
    points = [(-10, 0, 0), (10, 0, 0), (0, 1, 0)]
    sphere = minimal_enclosing_sphere(points)
    assert np.allclose(sphere.center, [0, 0, 0])
    assert np.isclose(sphere.radius, 10)
    print("Test 3 passed!")

    # Test 4: Four points
    points = [(5, 0, 1), (-1, -3, 4), (-1, -4, -3), (-1, 4, -3)]
    sphere = minimal_enclosing_sphere(points)
    assert np.allclose(sphere.center, [0, 0, 0])
    assert np.isclose(sphere.radius, np.sqrt(26))
    print("Test 4 passed!")

    print("All test cases passed!")

print("---------Question 1----------")
test_minimal_enclosing_sphere()


## Question 2:




print("---------Question 1----------")

