import Tasks

def test_task3():
    P = [(5, 0, 1), (-1, -3, 4), (-1, -4, -3), (-1, 4, -3)]
    L=[0,1,3,4,5,6,10]
    i:int = 0
    for l in L:
        print(f"------Test {i}: l={l}--------")
        print(Tasks.task3(P,l))
        i+1

#test_task3()
P = [(5, 0, 1), (-1, -3, 4), (-1, -4, -3), (-1, 4, -3)]

test_task3()

# print(Tasks.enum3(P)[0])
