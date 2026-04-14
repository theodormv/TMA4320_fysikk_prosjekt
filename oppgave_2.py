import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numba import jit



#oppgave a
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


#oppgave b
def MergeVectors(m1,m2,m3,m4):
    return np.concatenate((m1[:4],m2[:4],m3[:4],m4[:4],m1[4:],m2[4:],m3[4:],m4[4:]))


def SplitVectors(vec):
    vectors = np.zeros((4,8))

    vectors[0] = np.concatenate((vec[0 : 4], vec[16 : 20]))
    vectors[1] = np.concatenate((vec[4 : 8], vec[20 : 24]))
    vectors[2] = np.concatenate((vec[8 : 12], vec[24 : 28]))
    vectors[3] = np.concatenate((vec[12 : 16], vec[28 : 32]))

    return (vectors[0], vectors[1], vectors[2], vectors[3])


#oppgave c
def ConvertMatricesToVector(gamma, gammaThilde, omega, omegaThilde):
    return MergeVectors(C2x2MatrixToR8Vector(gamma), C2x2MatrixToR8Vector(gammaThilde), C2x2MatrixToR8Vector(omega), C2x2MatrixToR8Vector(omegaThilde))


def ConvertVectorToMatrices(vec):
    vecs = SplitVectors(vec)
    return (R8VectorToC2x2Matrix(vecs[0]),R8VectorToC2x2Matrix(vecs[1]),R8VectorToC2x2Matrix(vecs[2]),R8VectorToC2x2Matrix(vecs[3]))


#oppgave d,e,f
class SolveTools:
    delta = 0
    zeta = 0
    dimlessEnergy = 0
    dimlessLength = 0

    MaterialGammaLeft = np.zeros((2,2), dtype=np.complex128)
    MaterialGammaThildeLeft = np.zeros((2,2), dtype=np.complex128)
    MaterialGammaRight = np.zeros((2,2), dtype=np.complex128)
    MaterialGammaThildeRight = np.zeros((2,2), dtype=np.complex128)


    #oppgave d
    def CalculateN(gamma, gammaThilde):
        return np.linalg.inv(np.identity(2) - gamma @ gammaThilde)
    

    def CalculateNthilde(gamma, gammaThilde):
        return np.linalg.inv(np.identity(2) - gammaThilde @ gamma)
    

    def CalculateDelxVec(self, vec, dimlessEnergy):
        gamma, gammaThilde, omega, omegaThilde = ConvertVectorToMatrices(vec)

        delGamma = omega
        delGammaThilde = omegaThilde

        N = SolveTools.CalculateN(gamma, gammaThilde)
        Nthilde = SolveTools.CalculateNthilde(gamma, gammaThilde)

        delOmega = -2j*(dimlessEnergy + self.delta*1j)*gamma - 2*omega @ Nthilde @ gammaThilde @ omega
        delOmegaThilde = -2j*(dimlessEnergy + self.delta*1j)*gammaThilde - 2*omegaThilde @ N @ gamma @ omegaThilde

        return ConvertMatricesToVector(delGamma, delGammaThilde, delOmega, delOmegaThilde)


    #e
    def CalculateMdimDelxVecLOOP(self, xVec, mVec):
        delxmVec = np.zeros(mVec.shape)

        for i in range(mVec.shape[1]):
            delxmVec[:,i] = self.CalculateDelxVec(mVec[:,i], self.dimlessEnergy)

        return delxmVec


    #f
    def calculateBoundaryConditions(self, leftVec, rightVec):
        gammaLeft, gammaThildeLeft, omegaLeft, omegaThildeLeft = ConvertVectorToMatrices(leftVec)
        gammaRight, gammaThildeRight, omegaRight, omegaThildeRight = ConvertVectorToMatrices(rightVec)

        NL = SolveTools.CalculateN(self.MaterialGammaLeft, self.MaterialGammaThildeLeft)
        NLThilde = SolveTools.CalculateNthilde(self.MaterialGammaLeft, self.MaterialGammaThildeLeft)
        NR = SolveTools.CalculateN(self.MaterialGammaRight, self.MaterialGammaThildeRight)
        NRThilde = SolveTools.CalculateNthilde(self.MaterialGammaRight, self.MaterialGammaThildeRight)

        boundaryOmegaLeft = omegaLeft + 1/(self.zeta*self.dimlessLength) * (np.identity(2) - gammaLeft @ self.MaterialGammaThildeLeft) @ NL @ (self.MaterialGammaLeft - gammaLeft)
        boundaryOmegaThildeLeft = omegaThildeLeft + 1/(self.zeta*self.dimlessLength) * (np.identity(2) - gammaThildeLeft @ self.MaterialGammaLeft) @ NLThilde @ (self.MaterialGammaThildeLeft - gammaThildeLeft)

        boundaryOmegaRight = omegaRight - 1/(self.zeta*self.dimlessLength) * (np.identity(2) - gammaRight @ self.MaterialGammaThildeRight) @ NR @ (self.MaterialGammaRight - gammaRight)
        boundaryOmegaThildeRight = omegaThildeRight - 1/(self.zeta*self.dimlessLength) * (np.identity(2) - gammaThildeRight @ self.MaterialGammaRight) @ NRThilde @ (self.MaterialGammaThildeRight - gammaThildeRight)

        return ConvertMatricesToVector(boundaryOmegaLeft, boundaryOmegaThildeLeft, boundaryOmegaRight, boundaryOmegaThildeRight)

#g
def oppgave_2g(prb : SolveTools):
    prb.delta = 0.01
    prb.zeta = 3
    prb.dimlessEnergy = 1
    prb.dimlessLength = 1

    m = 101
    x = np.linspace(0, prb.dimlessLength, m)
    y = np.zeros((32,m))

    sols = []

    for en in range(3):
        prb.dimlessEnergy = en
        sol = sp.integrate.solve_bvp(prb.CalculateMdimDelxVecLOOP, prb.calculateBoundaryConditions, x, y)
        sols.append(sol)
        print(sol)

    return sols


#h
def CalculateGreensFunctions(sol):
    greens = np.zeros((sol.shape[1], 4, 4), dtype = np.complex128)

    for i in range(sol.shape[0]):
        gamma, gammaThilde, omega, omegaThilde = ConvertVectorToMatrices(sol[:,i])

        N = SolveTools.CalculateN(gamma, gammaThilde)
        NThilde = SolveTools.CalculateNthilde(gamma, gammaThilde)

        topLeft = 2*N - np.identity(2)
        topRight = 2*N @ gamma
        bottomLeft = -2*NThilde @ gammaThilde
        bottomRight = -2*NThilde + np.identity(2)

        greens[i] = np.concatenate ((np.concatenate((topLeft ,topRight), axis=1), np.concatenate((bottomLeft, bottomRight), axis = 1)), axis = 0)
    return greens


def CalculateDensityOfStates(greensFunctions): #greensfunctions is assumed to be (m x 4 x 4) tensor
    roHat3 = np.array(((1,0,0,0),(0,1,0,0),(0,0,-1,0),(0,0,0,-1)))
    print(greensFunctions.shape)
    D = np.zeros(greensFunctions.shape[0])
    for i in range(greensFunctions.shape[0]):
        D[i] = np.real(np.trace(roHat3 @ greensFunctions[i])) / 4
    return D


def oppgave_2h(prb : SolveTools):
    sols = np.array( oppgave_2g(prb) )
    x = np.linspace(0,1,101)
    
    fig = plt.figure()

    for i in range(len(sols)):
        greens = CalculateGreensFunctions(sols[i]["y"])
        print(greens[20])
        density = CalculateDensityOfStates( greens )
        ax = fig.add_subplot(1,3,i+1)
        ax.plot(x, density)
        ax.grid()
        fig.savefig("./output/default.png")
    
    fig.show()
        

#oppgave i
def nuPluss(prb : SolveTools):
    return np.arctanh(1/(prb.dimlessEnergy + prb.delta*1j))

def nuMinus(prb : SolveTools):
    return np.arctanh(-1/(prb.dimlessEnergy + prb.delta*1j))

def setRiccatiToTask2i(prb : SolveTools, phaseL = 0, phaseR = 0):
    plussElement = np.sinh(nuPluss(prb))/(1+np.cosh(nuPluss(prb)))
    minusElement = np.sinh(nuMinus(prb))/(1+np.cosh(nuMinus(prb)))

    prb.MaterialGammaLeft = np.array(((0, plussElement),(minusElement, 0)))*np.exp(phaseL*1j)
    prb.MaterialGammaThildeLeft = np.array(((0, minusElement),(plussElement, 0)))*np.exp(-phaseL*1j)

    prb.MaterialGammaRight = np.array(((0, plussElement),(minusElement, 0)))*np.exp(phaseR*1j)
    prb.MaterialGammaThildeRight = np.array(((0, minusElement),(plussElement, 0)))*np.exp(-phaseR*1j)


#oppgave j
def oppgave_2j():

    problem = SolveTools()
    problem.dimlessEnergy = 2
    problem.dimlessLength = 1
    problem.delta = 0.01
    problem.zeta = 3

    setRiccatiToTask2i(problem)

    m = 101
    x = np.linspace(0, problem.dimlessLength, m)
    y = np.zeros((32,m))

    sol = sp.integrate.solve_bvp(problem.CalculateMdimDelxVecLOOP, problem.calculateBoundaryConditions, x, y)
    greensFunctions = CalculateGreensFunctions(sol["y"])
    density = CalculateDensityOfStates(greensFunctions)
    plt.plot(x, density)
    plt.grid()
    plt.show()

    return sol



def oppgave_2():
    oppgave_2h(SolveTools())



if __name__ == "__main__":
    oppgave_2()