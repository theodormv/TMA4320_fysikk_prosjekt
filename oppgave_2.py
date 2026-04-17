import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm



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
def ConvertMatricesToVector(gamma, gammaTilde, omega, omegaTilde):
    return MergeVectors(C2x2MatrixToR8Vector(gamma), C2x2MatrixToR8Vector(gammaTilde), C2x2MatrixToR8Vector(omega), C2x2MatrixToR8Vector(omegaTilde))


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
    MaterialGammaTildeLeft = np.zeros((2,2), dtype=np.complex128)
    MaterialGammaRight = np.zeros((2,2), dtype=np.complex128)
    MaterialGammaTildeRight = np.zeros((2,2), dtype=np.complex128)


    #oppgave d
    def CalculateN(gamma, gammaTilde):
        return np.linalg.inv(np.identity(2) - gamma @ gammaTilde)
    

    def CalculateNTilde(gamma, gammaTilde):
        return np.linalg.inv(np.identity(2) - gammaTilde @ gamma)
    

    def CalculateDelxVec(self, vec, dimlessEnergy):
        gamma, gammaTilde, omega, omegaTilde = ConvertVectorToMatrices(vec)

        delGamma = omega
        delGammaTilde = omegaTilde

        N = SolveTools.CalculateN(gamma, gammaTilde)
        NTilde = SolveTools.CalculateNTilde(gamma, gammaTilde)

        delOmega = -2j*(dimlessEnergy + self.delta*1j)*gamma - 2*omega @ NTilde @ gammaTilde @ omega
        delOmegaTilde = -2j*(dimlessEnergy + self.delta*1j)*gammaTilde - 2*omegaTilde @ N @ gamma @ omegaTilde

        return ConvertMatricesToVector(delGamma, delGammaTilde, delOmega, delOmegaTilde)


    #e
    def CalculateMdimDelxVecLOOP(self, xVec, mVec):
        delxmVec = np.zeros(mVec.shape)

        for i in range(mVec.shape[1]):
            delxmVec[:,i] = self.CalculateDelxVec(mVec[:,i], self.dimlessEnergy)

        return delxmVec


    #f
    def calculateBoundaryConditions(self, leftVec, rightVec):
        gammaLeft, gammaTildeLeft, omegaLeft, omegaTildeLeft = ConvertVectorToMatrices(leftVec)
        gammaRight, gammaTildeRight, omegaRight, omegaTildeRight = ConvertVectorToMatrices(rightVec)

        NL = SolveTools.CalculateN(self.MaterialGammaLeft, self.MaterialGammaTildeLeft)
        NLTilde = SolveTools.CalculateNTilde(self.MaterialGammaLeft, self.MaterialGammaTildeLeft)
        NR = SolveTools.CalculateN(self.MaterialGammaRight, self.MaterialGammaTildeRight)
        NRTilde = SolveTools.CalculateNTilde(self.MaterialGammaRight, self.MaterialGammaTildeRight)

        boundaryOmegaLeft = omegaLeft + 1/(self.zeta*self.dimlessLength) * (np.identity(2) - gammaLeft @ self.MaterialGammaTildeLeft) @ NL @ (self.MaterialGammaLeft - gammaLeft)
        boundaryOmegaTildeLeft = omegaTildeLeft + 1/(self.zeta*self.dimlessLength) * (np.identity(2) - gammaTildeLeft @ self.MaterialGammaLeft) @ NLTilde @ (self.MaterialGammaTildeLeft - gammaTildeLeft)

        boundaryOmegaRight = omegaRight - 1/(self.zeta*self.dimlessLength) * (np.identity(2) - gammaRight @ self.MaterialGammaTildeRight) @ NR @ (self.MaterialGammaRight - gammaRight)
        boundaryOmegaTildeRight = omegaTildeRight - 1/(self.zeta*self.dimlessLength) * (np.identity(2) - gammaTildeRight @ self.MaterialGammaRight) @ NRTilde @ (self.MaterialGammaTildeRight - gammaTildeRight)

        return ConvertMatricesToVector(boundaryOmegaLeft, boundaryOmegaTildeLeft, boundaryOmegaRight, boundaryOmegaTildeRight)

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
    return sols


#h
def CalculateGreensFunctions(sol):
    greens = np.zeros((sol.shape[1], 4, 4), dtype = np.complex128)

    for i in range(sol.shape[1]):
        gamma, gammaTilde, omega, omegaTilde = ConvertVectorToMatrices(sol[:,i])

        N = SolveTools.CalculateN(gamma, gammaTilde)
        NTilde = SolveTools.CalculateNTilde(gamma, gammaTilde)

        topLeft = 2*N - np.identity(2)
        topRight = 2*N @ gamma
        bottomLeft = -2*NTilde @ gammaTilde
        bottomRight = -2*NTilde + np.identity(2)

        greens[i] = np.concatenate ((np.concatenate((topLeft ,topRight), axis=1), np.concatenate((bottomLeft, bottomRight), axis = 1)), axis = 0)

    return greens


def CalculateDensityOfStates(greensFunctions): #greensfunctions is assumed to be (m x 4 x 4) tensor
    roHat3 = np.array(((1,0,0,0),(0,1,0,0),(0,0,-1,0),(0,0,0,-1)))
    D = np.zeros(greensFunctions.shape[0])
    for i in range(greensFunctions.shape[0]):
        D[i] = np.real(np.einsum("ii", roHat3 @ greensFunctions[i])) / 4
    return D


def oppgave_2h(prb : SolveTools):
    sols = np.array( oppgave_2g(prb) )
    x = np.linspace(0,1,101)
    
    fig = plt.figure()

    ax = fig.add_subplot(1,1,1)
    ax.grid()
    styles = ["-", "--", "-."]

    for i in range(len(sols)):
        greens = CalculateGreensFunctions(sols[i]["y"])
        density = CalculateDensityOfStates( greens )
        ax.plot(x, density, styles[i], label=r"$\epsilon = $" + str(i) )
    ax.legend()
    fig.savefig("./output/default_oppg_h.png")

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
    prb.MaterialGammaTildeLeft = np.array(((0, minusElement),(plussElement, 0)))*np.exp(-phaseL*1j)

    prb.MaterialGammaRight = np.array(((0, plussElement),(minusElement, 0)))*np.exp(phaseR*1j)
    prb.MaterialGammaTildeRight = np.array(((0, minusElement),(plussElement, 0)))*np.exp(-phaseR*1j)


#oppgave j
def oppgave_2j():

    problem = SolveTools()
    problem.dimlessEnergy = 0.25
    problem.dimlessLength = 1
    problem.delta = 0.01
    problem.zeta = 3

    setRiccatiToTask2i(problem)

    m = 101
    x = np.linspace(0, problem.dimlessLength, m)
    y = np.zeros((32,m))

    sol = sp.integrate.solve_bvp(problem.CalculateMdimDelxVecLOOP, problem.calculateBoundaryConditions, x, y, max_)
    greensFunctions = CalculateGreensFunctions(sol["y"])
    density = CalculateDensityOfStates(greensFunctions)
    plt.plot(x, density)
    plt.grid()
    plt.savefig("./output/eps_0_25_oppg_j.png")
    plt.show()

    return sol


#oppgave k

"""Helper funtions"""
#h
def CalculateGreensFunction(sol):
    greens = np.zeros(( 4, 4), dtype = np.complex128)

    gamma, gammaTilde, omega, omegaTilde = ConvertVectorToMatrices(sol)

    N = SolveTools.CalculateN(gamma, gammaTilde)
    NTilde = SolveTools.CalculateNTilde(gamma, gammaTilde)

    topLeft = 2*N - np.identity(2)
    topRight = 2*N @ gamma
    bottomLeft = -2*NTilde @ gammaTilde
    bottomRight = -2*NTilde + np.identity(2)

    greens = np.concatenate ((np.concatenate((topLeft ,topRight), axis=1), np.concatenate((bottomLeft, bottomRight), axis = 1)), axis = 0)

    return greens
"""END helperfuntions"""

def oppgave_2k():

    epsilon = np.linspace(2,0,101)
    lengths = np.array((0.5,1,2))
    
    epsN = 101


    xm = 101

    fig = plt.figure()


    y = np.zeros((32,xm))
    
    ax = fig.add_subplot(1,1,1)
    ax.grid()
    ax.set_xlabel("Energy")
    ax.set_ylabel("DOS")

    for l in lengths:
        x = np.linspace(0,l,xm)
        DOS = np.zeros(epsN)
        sols = np.zeros((epsN, 32), dtype = np.complex128)
        i = 0
        for eps in tqdm(epsilon):
            problem = SolveTools()
            problem.delta = 0.01
            problem.zeta = 3
            problem.dimlessLength = l
            problem.dimlessEnergy = eps

            setRiccatiToTask2i(problem)

            sol = sp.integrate.solve_bvp(problem.CalculateMdimDelxVecLOOP, problem.calculateBoundaryConditions, x, y, max_nodes = xm)
            y = sol["y"]
            solAtX_2 = sol["y"][:, ((xm+1)//2)]
            sols[i] = solAtX_2
            i += 1

        DOS = CalculateDensityOfStates( CalculateGreensFunctions(sols) )

        ax.plot(epsilon, DOS, label = r"$l =$" + str(float(l)))
    
    ax.legend()
    fig.suptitle(r"DOS at $x = \frac{l}{2}$")
    

    fig.tight_layout()
    fig.savefig("./output/default_oppg_k.png")
    fig.show()

#l
def Commutator(A,B):
    return A @ B - B @ A


def CalculateCurrentIntegrand(greens):
    roHat3 = np.array(((1,0,0,0),(0,1,0,0),(0,0,-1,0),(0,0,0,-1)))
    return np.real(np.einsum("ii", roHat3 @ Commutator(greens, ) ))


def oppgave_2l():
    pass


def oppgave_2():
    oppgave_2j()

if __name__ == "__main__":
    oppgave_2()
