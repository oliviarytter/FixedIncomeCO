import numpy as np

# Loops
for item in range(0,5):
    pass
    # print(item)

# Lists
L = []
L = 5*[None]
L = [i for i in range(0,5)]
L1, L2 = [1,7,3], [4,5,9]
L3 = L1 + L2
print(dir(L3))
L4 = L3.sort()
print(L3)
print(L4)

# Exponentials and logs in numpy
x = np.exp(1)
y = np.log(x)

# Arrays
v = np.zeros([5])
v = np.ones([5])
v = np.array([1,2,3,4,5])
v1 = v[0:]
v2 = v[1:2]
v3 = v[1:5]
v4 = v[-3:]
v5 = v[-3:-1]

# Matrices
M = np.zeros([5,5])
M = np.ones([5,5])
M = np.eye(5)
M[1][1] = 4
M[3,4] = 2
M = np.array([[1,2,3,4,5],[5,6,7,8,9],[10,11,12,13,14],[11,12,13,14,15],[16,17,18,19,20]])
dims = M.shape
rank = np.linalg.matrix_rank(M)
M1 = M*v
M2 = np.matmul(M,v)
M3 = np.transpose(M)
M4 = np.hstack([M,M])
M5 = np.vstack([M,M])
M6 = np.array([[-1,2,3,4,5],[5,-6,7,8,9],[10,11,-12,13,14],[11,12,13,14,15],[16,17,18,19,20]])
v6 = np.linalg.solve(M6,v)
