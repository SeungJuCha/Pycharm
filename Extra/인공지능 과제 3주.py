import numpy as np

A = np.array([[1,2,3],[4,5,6]])

print('Split by row = ',np.split(A,2, axis = 0),end = '\n')
print(np.vsplit(A,2))

print('Split by column = ','\n',np.split(A,3, axis = 1))
print(np.hsplit(A,3))
"""3x + 2y + z = 1
   7x - y +3z = 1
   5x + 4y -2z = 1"""
def finding_numerator_A (A,B):
    Det_A_B = []
    for i in range(len(A)):
        A_1 = A.copy()
        for j in range(len(A_1)):
            A_1[j][i] = B[i]
            """changing A_1 with B element"""
        det_A = np.linalg.det(A_1)
        """calculating determinant of changed A_1"""
        Det_A_B.append(det_A)
        i += 1
        if i < len(A):
            continue
        else :
            break
    return Det_A_B

A = np.array([[3, 2, 1],
               [7, -1, 3],
               [5, 4, -2]])

B = np.array([1,1,1])
Det_A  = np.linalg.det(A)
Answer = finding_numerator_A(A,B)/Det_A
print('Answer =',Answer)

#2-2
coef_arr = np.array([[3,2,1],[1,-1,3],[5,4,-2]])
const_arr = np.array([7,3,1])

coef_arr_inv = np.linalg.inv(coef_arr)


x_arr = np.dot(coef_arr_inv, const_arr)
print(x_arr)
print(f'x1 = {x_arr[0]}, x2 = {x_arr[1]}, x3 = {x_arr[2]}')






