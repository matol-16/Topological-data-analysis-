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
def test_task1():
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



## Question 2:

#Ici, on déduit que la filtration value de chaque simplexe est le MEB (à vérifier)

def task2(points,emu):
    """Compute the filtration value for the points in emu"""
    points_chosen = [points[i] for i in emu]
    filtration = minimal_enclosing_sphere(points_chosen).radius

    return filtration

# Test cases
def test_task2():
        P = [(5, 0, 1), (-1, -3, 4), (-1, -4, -3), (-1, 4, -3)]

        enu=[0]
        assert np.allclose(task2(P,enu), 0)  
        print(f"Test({enu})passed!")

        enu=[1]
        assert np.allclose(task2(P,enu), 0)  
        print(f"Test({enu})passed!")

        enu=[2]
        assert np.allclose(task2(P,enu), 0)  
        print(f"Test({enu})passed!")

        enu=[3]
        assert np.allclose(task2(P,enu), 0)
        print(f"Test({enu})passed!")

        enu=[2,1]
        assert np.allclose(task2(P,enu), 3.53553)   
        print(f"Test({enu})passed!")

        enu=[1,0]
        assert np.allclose(task2(P,enu), 3.67425)   
        print(f"Test({enu})passed!")

        enu=[3,2]
        assert np.allclose(task2(P,enu), 4)   
        print(f"Test({enu})passed!")

        enu=[2,0]
        assert np.allclose(task2(P,enu), 4.12311)   
        print(f"Test({enu})passed!")

        enu=[3,0]
        assert np.allclose(task2(P,enu), 4.12311)   
        print(f"Test({enu})passed!")

        enu=[2,1,0]
        assert np.allclose(task2(P,enu), 4.39525)   
        print(f"Test({enu})passed!")

        enu=[3,2,0]
        assert np.allclose(task2(P,enu), 4.71495)   
        print(f"Test({enu})passed!")

        enu=[3,1]
        assert np.allclose(task2(P,enu), 4.94975)   
        print(f"Test({enu})passed!")

        enu=[3,2,1]
        assert np.allclose(task2(P,enu), 5)   
        print(f"Test({enu})passed!")

        enu=[3,1,0]
        assert np.allclose(task2(P,enu), 5.04975)   
        print(f"Test({enu})passed!")

        enu=[3,2,1,0]
        assert np.allclose(task2(P,enu), 5.09902)   
        print(f"Test({enu})passed!")

        print("Test 2 all passed! ")


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

def enum3_mathias(points):
    "crée le tableau énumérant pour chaque taille (lignes) les sous ensembles: il représente un arbre"
    "il doit permettre de savoir si un simplexe de taille inférieure à la taille considérée existe"
    n=len(points)
    enum=[]
    l1=[ens for ens in range(n)]

    enum.append([(ens,) for ens in range(n)])

    k=2 #taille du sous ensemble
    while(k<=n):
        enum.append([comb for comb in combinations(l1, k)])#on va exploiter le format de sortie de la fonction combinaisons
        k=k+1
    return enum

  
def task3_mathias(points,l):
    """implement an algorithm that enumerates the simplexes and their filtration values."""

    emu = enum3(points)
    simplex={(0):0} #indique 
    #IsSimplex = {0:True} #dictionnaire indiquant si un sous ensemble forme un simplexe
    IsSimplex= [[0] * len(sublist) for sublist in emu] # le tableau de suivi: 0 si jamais vu, 1 si simplexe, 2 si pas un simplexe

    n=len(points)


    #Vestion 1 Mathias: pas optimale, on évite pas tj de calculer les simplexes qui en contiennent d'autres
    for i in range(len(emu)):
        for j in range(len(emu[i])):
            test = IsSimplex[i][j]
            if(test==0): #pas encore connu
                filtration = task2(points,emu[i][j])
                if (filtration > l): #pas un simplexe
                    emu[i][j] = 2
                    if(i<n-1): #on s'assure qu'on est pas à la dernière ligne
                        for k in range(n-i-1-j):
                         emu[i+1][j+k]=2 #les sous ensembles de taille supérieure qui contiennent emu[i,j] ne sont pas non plus des simplexes
                else:
                    emu[i][j] = 1
                    simplex[emu[i][j]]=filtration
            elif(test==1): #est un simplexe -> normalement impossible de revenir sur nos pas !
                raise RecursionError("l'ago revient sur ses pas !")
            #Sinon, test ==2 et ce n'est pas un simplexe. on passe au prochain. Pas besoin de tester cette éventualité
    return simplex   

 
def task3(points,l):
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

def test_task3():
    P = [(5, 0, 1), (-1, -3, 4), (-1, -4, -3), (-1, 4, -3)]
    task3(P,1000)

test_task3()

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
        


    

