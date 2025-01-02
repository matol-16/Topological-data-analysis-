import Tasks as t
import numpy as np

# Test cases
def test_task1():
    """Test cases for minimal enclosing sphere."""
    # Test 1: Single point
    points = [(0, 0, 0)]
    sphere = t.minimal_enclosing_sphere(points)
    assert np.allclose(sphere.center, [0, 0, 0])
    assert np.isclose(sphere.radius, 0)
    print("Test 1 passed!")

    # Test 2: Two points
    points = [(0, 0, 0), (2, 0, 0)]
    sphere = t.minimal_enclosing_sphere(points)
    assert np.allclose(sphere.center, [1, 0, 0])
    assert np.isclose(sphere.radius, 1)
    print("Test 2 passed!")

    # Test 3: Three points
    points = [(-10, 0, 0), (10, 0, 0), (0, 1, 0)]
    sphere = t.minimal_enclosing_sphere(points)
    assert np.allclose(sphere.center, [0, 0, 0])
    assert np.isclose(sphere.radius, 10)
    print("Test 3 passed!")

    # Test 4: Four points
    points = [(5, 0, 1), (-1, -3, 4), (-1, -4, -3), (-1, 4, -3)]
    sphere = t.minimal_enclosing_sphere(points)
    assert np.allclose(sphere.center, [0, 0, 0])
    assert np.isclose(sphere.radius, np.sqrt(26))
    print("Test 4 passed!")

 

    print("All test cases passed!")


# Test cases
def test_task2():
        P = [(5, 0, 1), (-1, -3, 4), (-1, -4, -3), (-1, 4, -3)]

        enu=[0]
        assert np.allclose(t.task2(P,enu), 0)  
        print(f"Test({enu})passed!")

        enu=[1]
        assert np.allclose(t.task2(P,enu), 0)  
        print(f"Test({enu})passed!")

        enu=[2]
        assert np.allclose(t.task2(P,enu), 0)  
        print(f"Test({enu})passed!")

        enu=[3]
        assert np.allclose(t.task2(P,enu), 0)
        print(f"Test({enu})passed!")

        enu=[2,1]
        assert np.allclose(t.task2(P,enu), 3.53553)   
        print(f"Test({enu})passed!")

        enu=[1,0]
        assert np.allclose(t.task2(P,enu), 3.67425)   
        print(f"Test({enu})passed!")

        enu=[3,2]
        assert np.allclose(t.task2(P,enu), 4)   
        print(f"Test({enu})passed!")

        enu=[2,0]
        assert np.allclose(t.task2(P,enu), 4.12311)   
        print(f"Test({enu})passed!")

        enu=[3,0]
        assert np.allclose(t.task2(P,enu), 4.12311)   
        print(f"Test({enu})passed!")

        enu=[2,1,0]
        assert np.allclose(t.task2(P,enu), 4.39525)   
        print(f"Test({enu})passed!")

        enu=[3,2,0]
        assert np.allclose(t.task2(P,enu), 4.71495)   
        print(f"Test({enu})passed!")

        enu=[3,1]
        assert np.allclose(t.task2(P,enu), 4.94975)   
        print(f"Test({enu})passed!")

        enu=[3,2,1]
        assert np.allclose(t.task2(P,enu), 5)   
        print(f"Test({enu})passed!")

        enu=[3,1,0]
        assert np.allclose(t.task2(P,enu), 5.04975)   
        print(f"Test({enu})passed!")

        enu=[3,2,1,0]
        assert np.allclose(t.task2(P,enu), 5.09902)   
        print(f"Test({enu})passed!")

        print("Test 2 all passed! ")
def test_task3():
    P = [(5, 0, 1), (-1, -3, 4), (-1, -4, -3), (-1, 4, -3)]
    t.task3(P,1000)


print("---------Question 1------------")
test_task1()
print("---------Question 2------------")
test_task2()
print("---------Question 3------------")
test_task3()

