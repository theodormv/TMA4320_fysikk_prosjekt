from oppg2.base_funcs import *
import theo_oppg2 as theo
from jax import random


#C2x2MatrixToR8Vector


def C2x2MatrixToR8Vector(matrix):
    m, _, _ = matrix.shape
    matrix = matrix.reshape(4, m)
    real = jnp.real(matrix)
    imag = jnp.imag(matrix)

    return jnp.concat([real, imag], axis = 0)


#R8VectorToC2x2Matrix
def R8VectorToC2x2MatrixTEST(vec): #vec.shape = 8 x m
    entry, m = vec.shape
    matrix = jnp.reshape(vec, (m,2,2,2))
    matrix = matrix[:, 0, :, :] + 1j* matrix[:, 1, :, :]

    return matrix

def flatten_matrices_fast(matrix_list):
    complete_matrix = jnp.stack(matrix_list, axis=0)
    real_part = jnp.real(complete_matrix)
    imaginary_part = jnp.imag(complete_matrix)

    real_transform = jnp.stack([real_part, imaginary_part], axis=1).reshape(32, 1)
    return real_transform




matrix_list = list()

for i in range(4):
    key = random.key(42 + i)
    key2 = random.key(41 + 5*i)
    vecreal = random.randint(key,dtype=jnp.int32, shape=(2, 2), minval=0, maxval=20)
    vecimag = 1j*random.randint(key2,dtype=jnp.int32, shape=(2, 2), minval=0, maxval=20)
    vec = vecreal + vecimag
    matrix_list.append(vec)

print(matrix_list[0])
A = flatten_matrices_fast(matrix_list)

B = A.reshape(-1, 2, 2, 2)

print(B[:, 0] + B[:, 1]*1j)

'''
print(vec, '\n\n\n\n')
tempvec = C2x2MatrixToR8Vector(vec)
print(tempvec)
#print(R8VectorToC2x2MatrixTEST(tempvec))
'''
