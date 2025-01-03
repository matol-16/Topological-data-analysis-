import random
import numpy as np
from scipy.linalg import null_space
from scipy.spatial import ConvexHull


## QUESTION 1:

class Sphere:
    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = radius

    def contains(self, point):
        """Check if the sphere contains the given point."""
        return np.linalg.norm(self.center - np.array(point)) <= self.radius #+ 1e-9
    
    def contains_strict(self,point):
        """Check if the sphere contains the given point."""
        norme = np.linalg.norm(self.center - np.array(point))
        return not (np.isclose((norme),self.radius) or norme>self.radius+1e-9)
    
    def onradius(self,point):
        """Check if the point is on the sphere"""
        return np.isclose(np.linalg.norm(self.center - np.array(point)),self.radius)

def make_sphere_n_points(points):
    """
    trouver le minimal circumcircle dans l'espace d-dimension de n points (n <= d+1)
    
    parametre :
        points: np.ndarray, shape (n, d)
    return:
        Sphere
    """
    points = np.array(points)
    print(points)
    n, d = points.shape
    
    #构造n-1个向量
    A = points[1:] - points[0]
    print(A)
    
    #计算中点
    m=(points[1:] + points[0])*0.5
    print(m)
    #计算b
    b=[]
    for i in range(n-1):
        b.append(np.dot(A[i],m[i]))
    print(b)
    
    #在A张成的线性子空间中解Ax=b,得到圆心
    if n==d+1:
        center =  np.linalg.solve(A@A, b) @ A 
    else:
        center = np.linalg.lstsq(A@A, b, rcond=None)[0] @ A
    print(center)
    #计算半径
    radius = np.linalg.norm(center - points[0])
    
    # 返回最小外接球的实例
    return Sphere(center, radius)
    
def trivial(R):
    
    """Find the minimal sphere for 0, 1, 2, 3, or 4 points."""
    if not R:
        return Sphere([0, 0, 0], 0)
    elif len(R) == 1:
        return Sphere(R[0], 0)
    elif len(R) >= 1:
        return make_sphere_n_points(R)

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
  
def task3_mathias(points,l):
    """implement an algorithm that enumerates the simplexes and their filtration values."""

    emu = enum3(points)
    simplex = {tuple([i]): 0 for i in range(len(points))} #on initialise le premier simplexe
    #IsSimplex = {0:True} #dictionnaire indiquant si un sous ensemble forme un simplexe
    IsSimplex= [[0] * len(sublist) for sublist in emu] # le tableau de suivi: 0 si jamais vu, 1 si simplexe, 2 si pas un simplexe

    n=len(points)


    #Vestion 1 Mathias: pas optimale, on évite pas tj de calculer les simplexes qui en contiennent d'autres
    for i in range(1,len(emu)):
        for j in range(len(emu[i])):
            test = IsSimplex[i][j]
            if(test==0): #pas encore connu
                filtration = task2(points,emu[i][j])
                if (filtration > l): #pas un simplexe
                    IsSimplex[i][j] = 2
                    if(i<n-1): #on s'assure qu'on est pas à la dernière ligne
                        for k in range(n-i-1-j):
                         IsSimplex[i+1][j+k]=2 #les sous ensembles de taille supérieure qui contiennent emu[i,j] ne sont pas non plus des simplexes
                else:
                    IsSimplex[i][j] = 1
                    simplex[tuple(emu[i][j])]=filtration
            elif(test==1): #est un simplexe -> normalement impossible de revenir sur nos pas !
                raise RecursionError("l'ago revient sur ses pas !")
            #Sinon, test ==2 et ce n'est pas un simplexe. on passe au prochain. Pas besoin de tester cette éventualité
    
    for key, value in simplex.items():
        print(f"{key} -> {value}")
    
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

def Is_in_alpha_complex(P):
    R=[]
    if len(P)<len(P[0])+1  :
        return True
    
    for i in range(len(P[0])+1):
       R.append(P.pop(random.randint(0, len(P) - 1)))

    MEB=make_sphere_n_points(R)
    
    for p in R :
       if not MEB.onradius(p):
           return False
    
    for p in P:
       if MEB.contains(p):
           return False

    return True

def task4(points):
    """"Reuse the LP-type algorithm with new parameters in order to determine
if a simplex is in the α-complex and its filtration value. Note that this is less
standard than for the MEB, you need to explain how this new problem fits in
the framework."""

    """On part du principe que pour trouver le cercle le plus petit possible qui a ces points sur sa frontière,
    il suffit de calculer leur MEB (qui a nécessairement 2 points sur sa frontière) et de voir si le MEB a tous les points sur sa frontière"""
    
    return Is_in_alpha_complex(points)


# Test cases
def test_task1():
    """Test cases for minimal enclosing sphere."""
    # Test 1: Single point
    points = [(0, 0, 0)]
    sphere = minimal_enclosing_sphere(points)
    assert np.allclose(sphere.center, [0, 0, 0])
    assert np.isclose(sphere.radius, 0)
    print("Test 1.1 passed!")

    # Test 2: Two points
    points = [(0, 0, 0), (2, 0, 0)]
    sphere = minimal_enclosing_sphere(points)
    print(sphere.center)
    assert np.allclose(sphere.center, [1, 0, 0])
    assert np.isclose(sphere.radius, 1)
    print("Test 1.2 passed!")

    # Test 3: Three points
    points = [(-10, 0, 0), (10, 0, 0), (0, 1, 0)]
    sphere = minimal_enclosing_sphere(points)
    assert np.allclose(sphere.center, [0, 0, 0])
    assert np.isclose(sphere.radius, 10)
    print("Test 1.3 passed!")

    # Test 4: Four points
    points = [(5, 0, 1), (-1, -3, 4), (-1, -4, -3), (-1, 4, -3)]
    sphere =  minimal_enclosing_sphere(points)
    assert np.allclose(sphere.center, [0, 0, 0])
    assert np.isclose(sphere.radius, np.sqrt(26))
    print("Test 1.4 passed!")

 

    print("All test cases passed!")


# Test cases
def test_task2():
        P = [(5, 0, 1), (-1, -3, 4), (-1, -4, -3), (-1, 4, -3)]

        enu=[0]
        assert np.allclose(task2(P,enu), 0)  
        print(f"Test({enu})passed!")

        enu=[1]
        assert np.allclose( task2(P,enu), 0)  
        print(f"Test({enu})passed!")

        enu=[2]
        assert np.allclose( task2(P,enu), 0)  
        print(f"Test({enu})passed!")

        enu=[3]
        assert np.allclose( task2(P,enu), 0)
        print(f"Test({enu})passed!")

        enu=[2,1]
        assert np.allclose( task2(P,enu), 3.53553)   
        print(f"Test({enu})passed!")

        enu=[1,0]
        assert np.allclose( task2(P,enu), 3.67425)   
        print(f"Test({enu})passed!")

        enu=[3,2]
        assert np.allclose( task2(P,enu), 4)   
        print(f"Test({enu})passed!")

        enu=[2,0]
        assert np.allclose( task2(P,enu), 4.12311)   
        print(f"Test({enu})passed!")

        enu=[3,0]
        assert np.allclose( task2(P,enu), 4.12311)   
        print(f"Test({enu})passed!")

        enu=[2,1,0]
        assert np.allclose( task2(P,enu), 4.39525)   
        print(f"Test({enu})passed!")

        enu=[3,2,0]
        assert np.allclose( task2(P,enu), 4.71495)   
        print(f"Test({enu})passed!")

        enu=[3,1]
        assert np.allclose( task2(P,enu), 4.94975)   
        print(f"Test({enu})passed!")

        enu=[3,2,1]
        assert np.allclose( task2(P,enu), 5)   
        print(f"Test({enu})passed!")

        enu=[3,1,0]
        assert np.allclose( task2(P,enu), 5.04975)   
        print(f"Test({enu})passed!")

        enu=[3,2,1,0]
        assert np.allclose( task2(P,enu), 5.09902)   
        print(f"Test({enu})passed!")

        print("Test 2 all passed! ")
        
def test_task3():
    P = [(5, 0, 1), (-1, -3, 4), (-1, -4, -3), (-1, 4, -3)]
    task3(P,1000)


def test_task4():
    P=[(0,5,0),(3,4,0),(-3,4,0)]
    
    
    print(f"---- Test for {P}")
    a= task4(P)
    print(f"Complex ? {a}")

    P.append((0,0,4))
    print(f"---- Test for {P}")
    a= task4(P)
    print(f"Complex ? {a}")

    P.append((0,0,-4))
    print(f"---- Test for {P}")
    a= task4(P)
    print(f"Complex ? {a}")





print("---------Question 1------------")
test_task1()
print("---------Question 2------------")
test_task2()
print("---------Question 3------------")
test_task3()
#print("fonction mathias:")
#test_task3_mathias()
print("---------Question 4------------")
test_task4()


        

    

