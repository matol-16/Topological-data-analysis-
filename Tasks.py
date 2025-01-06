import random
import numpy as np

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
    Trouver le minimal circumcircle dans l'espace d-dimension de n points (n <= d+1).
    
    Paramètres :
        points: np.ndarray, shape (n, d)
    Retourne :
        Sphere
    """
    points = np.array(points, dtype=float)
    
    # Calcul de la matrice A et du vecteur b
    diffs = points[1:] - points[0]  # Différences (P^n - P^0) pour n = 1, ..., N-1
    A = 2 * np.dot(diffs, diffs.T)  # Matrice A
    b = np.sum(diffs ** 2, axis=1)  # Vecteur b

    # Résolution du système linéaire pour trouver les coefficients k
    k = np.linalg.solve(A, b)

    # Calculer le centre
    center = points[0] + np.dot(k, diffs)

    # Calculer le rayon
    radius = np.linalg.norm(center - points[0])

    # Retourner une instance de Sphere
    return Sphere(center=center, radius=radius)
    
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

def Is_in_alpha_complex(P,R):
    """Check if the simplex R is in the alpha complex P"""
    
    #On suppose que le simplex est un simplexe
    if not R or len(R)<=len(P[0])+1  :
        return True

    MEB=make_sphere_n_points(R)
    
    for p in P:
       if MEB.contains(p):
           return False

    return True

def filtration_value(R):
    """Compute the filtration value of the simplex R"""
    return make_sphere_n_points(R).radius

def task4_Is_in_alpha_complex(points,R):
    """"Reuse the LP-type algorithm with new parameters in order to determine
if a simplex is in the α-complex and its filtration value. Note that this is less
standard than for the MEB, you need to explain how this new problem fits in
the framework."""

    """On part du principe que pour trouver le cercle le plus petit possible qui a ces points sur sa frontière,
    il suffit de calculer leur MEB (qui a nécessairement 2 points sur sa frontière) et de voir si le MEB a tous les points sur sa frontière"""
    
    return Is_in_alpha_complex(points,R)

def task4_filtration_value(R):
    """"Reuse the LP-type algorithm with new parameters in order to determine
if a simplex is in the α-complex and its filtration value. Note that this is less
standard than for the MEB, you need to explain how this new problem fits in
the framework."""

    """On part du principe que pour trouver le cercle le plus petit possible qui a ces points sur sa frontière,
    il suffit de calculer leur MEB (qui a nécessairement 2 points sur sa frontière) et de voir si le MEB a tous les points sur sa frontière"""
    
    return filtration_value(R)

def task5(points,K,l):
      """ Given a set P of n points in Rd, implement an algorithm that enu-
merates the simplexes of dimension at most k and filtration value at most l of
the α-complex and their filtration values.""" 
      enum = enum3(points)
      print(f"enum={enum}")
      filtration_value=0
      IsSimplex = {tuple([i]): 1 for i in range(len(points))}

      simplex = {tuple([i]): Sphere(points[i], 0) for i in range(len(points))} #on initialise le premier simplexe

      for i in range(1,K):
          for j in range(len(enum[i-1])):
              
              current_simplex = enum[i-1][j]

              for k in range(len(points)):
                  
                  if k in current_simplex:
                      IsSimplex[tuple(current_simplex + [k])] = 0
                      break
                  
                  pn =tuple(current_simplex + [k])

                  if not Is_in_alpha_complex(points,pn):
                    IsSimplex[pn] = 0
                    break

                  if pn in simplex:
                    break

                  new_simplex = [points[idx] for idx in current_simplex] + [points[k]]

                  MEB = trivial(new_simplex)

                  if MEB.radius < l:
                      if MEB.radius > filtration_value:
                        filtration_value=MEB.radius
                      simplex[pn] = MEB
                      IsSimplex[pn] = 1
                  else:
                      IsSimplex[pn] = 0

                  
                  
                  print(f"new simplex: {pn} -> {MEB.radius} with filtration value {filtration_value}")

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
    R=P
    print(f"---- Test for {P}")
    a= task4_Is_in_alpha_complex(P,R)
    print(f"Complex ? {a}")
    print(f"filtration value: {task4_filtration_value(R)}")

    P.append((0,0,4))
    R=P
    print(f"---- Test for {P}")
    a= task4_Is_in_alpha_complex(P,R)
    print(f"Complex ? {a}")
    print(f"filtration value: {task4_filtration_value(R)}")

    P.append((0,0,-4))
    print(f"---- Test for {P}")
    a= task4_Is_in_alpha_complex(P,R)
    print(f"Complex ? {a}")
    print(f"filtration value: {task4_filtration_value(R)}")

def test_task5():
 #generate random n points in R^d:
 n=random.randint(5,20)
 print(f"n={n}")
 d=random.randint(2,5)
 print(f"d={d}")
 points=[tuple(np.random.rand(d)) for i in range(n)]
 print(f"Points: {points}")
 k=random.randint(2,d)
 print(f"k={k}")
 l=random.randint(1,10)
 print(f"l={l}")
 task5(points,k,l)

#print("---------Question 1------------")
#test_task1()
#print("---------Question 2------------")
#test_task2()
#print("---------Question 3------------")
#test_task3()
#print("fonction mathias:")
#test_task3_mathias()
#print("---------Question 4------------")
#test_task4()
print("---------Question 5------------")
test_task5()    