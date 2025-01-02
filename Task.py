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




## Question 2:

#Ici, on déduit que la filtration value de chaque simplexe est le MEB (à vérifier)

def task2(points,emu):
    """Compute the filtration value for the points in emu"""
    points_chosen = [points[i] for i in emu]
    filtration = minimal_enclosing_sphere(points_chosen).radius

    return filtration

def enum_simplex2(points):
    "énumère et affiche les simplexes avec la valeur de filtrage"
    parties= [] #on fait la liste des sous ensembles de points:


    i, imax = 0, 2**len(points)-1 #on construit un itérateur i, dont les bits 1 seront les points sélectionnés dans le sous ensemble
    while i <= imax:
        s = []
        j, jmax = 0, len(points)-1
        while j <= jmax:
            if (i>>j)&1 == 1:
                s.append(points[j])
            j += 1
        parties.append(s)
        i += 1 

    #on affiche les simplexes avec filtration value

    for enum in parties:
        filtration = task2(points,enum)
        print(f"({enum}) -> {filtration}")






from itertools import combinations #On utilise ici une bibliothèque pour le travail combinatoire -> à faire à la main plus tard.

def enum3(points):
    """
    Génère un tableau où chaque ligne correspond aux sous-ensembles d'une certaine taille.
    """
    n = len(points)
    return [[list(comb) for comb in combinations(range(n), k)] for k in range(1, n + 1)]


  
def task3(points,l):
    """implement an algorithm that enumerates the simplexes and their filtration values."""
    
    
    enum = enum3(points)
    IsSimplex = {tuple([i]): 1 for i in range(len(points))}

    simplex = {tuple([i]): Sphere(points[i], 0) for i in range(len(points))} #on initialise le premier simplexe

    
    for i in range(1,len(enum)):
        for j in range(len(enum[i-1])):
            current_simplex = enum[i-1][j]

            for k in range(len(points)):

                pn =tuple(current_simplex + [k])

                if k in current_simplex:
                    IsSimplex[pn] = 0
                    break

                if pn in simplex:
                    break

                new_points = [points[idx] for idx in current_simplex] + [points[k]]

                MEB = minimal_enclosing_sphere(new_points)

                if MEB.radius < l:
                    simplex[pn] = MEB
                    IsSimplex[pn] = 1
                else:
                    IsSimplex[pn] = 0

    for key, value in simplex.items():
        print(f"{key} -> {value.radius}")



def task4(points):
    """"Reuse the LP-type algorithm with new parameters in order to determine
if a simplex is in the α-complex and its filtration value. Note that this is less
standard than for the MEB, you need to explain how this new problem fits in
the framework."""
    MEB = minimal_enclosing_sphere(points)
    for p in points:
        if not MEB.contains(p):
            return False
    return True
        


