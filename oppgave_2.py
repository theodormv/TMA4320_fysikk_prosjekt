import numpy as np
import scipy as sp

def C2x2MatrixToR8Vector(matrix):
    real = np.real(matrix)
    imag = np.imag(matrix)

    realVector = real.flatten()
    imagVector = imag.flatten()

    return np.array((realVector, imagVector)).flatten()

def R8VectorToC2x2Matrix(vec):
    real = vec[:4]
    imag = vec[4:]

    matrix = real.reshape((2,2)) + imag.reshape((2,2))*1j

    return matrix



def MergeVectors(m1,m2,m3,m4):
    return np.concatenate((m1[:4],m2[:4],m3[:4],m4[:4],m1[4:],m2[4:],m3[4:],m4[4:]))

def SplitVectors(vec):
    vectors = np.zeros((4,8))

    for i in range(0,4):
        vectors[i] = np.concatenate((vec[4*i : 4*i + 4], vec[16 + 4*i : 16 + 4*i + 4]))

    return (vectors[0], vectors[1], vectors[2], vectors[3])

"""

mat1 = np.array(((1,2),(3,4))) + np.array(((5,6),(7,8)))*1j
mat2 = mat1 + np.ones((2,2))*(10+10j)
mat3 = mat2 + np.ones((2,2))*(10+10j)
mat4 = mat3 + np.ones((2,2))*(10+10j)


VecMat1 = C2x2MatrixToR8Vector(mat1)
VecMat2 = C2x2MatrixToR8Vector(mat2)
VecMat3 = C2x2MatrixToR8Vector(mat3)
VecMat4 = C2x2MatrixToR8Vector(mat4)

print("split 2x2matrix", VecMat1)
print("marged R8 vector", R8VectorToC2x2Matrix(VecMat1))

print("merged Vectors", MergeVectors(VecMat1,VecMat2,VecMat3,VecMat4))
print("split merged matrix", SplitVectors(MergeVectors(VecMat1,VecMat2,VecMat3,VecMat4)))

"""