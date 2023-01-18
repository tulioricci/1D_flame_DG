import numpy as np
import matplotlib.pyplot as plt
import scipy.special

def mapping(x,LGL):
  
  #Face points
  xF = np.vstack((x[0:-1],x[1:  ]))

  #Internal points
  Ng = x.shape[0] - 1
  Np = LGL.shape[0]
  xmap = np.zeros((Np,Ng))
  for i in range(0,Ng):
    xmap[:,i] = xF[0,i] + 0.5*(1.0+LGL[:])*(xF[1,i] - xF[0,i])
       
  return xmap, xF
  
def normal_vectors(xF):

  nhat = xF*0.0
  nhat[0,:] = -1.0
  nhat[1,:] = 1.0
  
  return nhat

"""
"""
def LegendrePolynomials(N):

  #Normalized Legrendre polynomials
  phi = []
  for i in range(0,N):
    phi.append(scipy.special.legendre(i)/np.sqrt(2.0/(2.0*i + 1.0)))
   
  return phi

"""
"""
def GaussLobatto(N,phi):

  #The Legendre-Gauss-Lobatto are the roots of the
  #first derivative of the Legendre polynomials

  Np = N - 1

  # differentiation of a polynomial
  dphi = np.zeros((Np))
  for i in range(0,Np):
    dphi[i] = (Np-i)*phi[Np][Np-i]

  LGL = np.zeros((N))
  LGL[0] = -1.0
  LGL[1:-1] = np.roots(dphi)
  LGL[-1] = 1.0

  for j in range(len(LGL)-1,0,-1):
    for i in range(j):
      if LGL[i]> LGL[i+1]:
        temp = LGL[i]
        LGL[i] = LGL[i+1]
        LGL[i+1] = temp
      
  return LGL
  
"""
"""
def JacobiP(LGL,alpha,beta,ii,N):

  #I had to translate Hesthaven`s code due to normalization of the
  #Jacobi polynomials

  PL = np.zeros((N+1,len(LGL)))

  gamma0 = 2**(alpha+beta+1)/(alpha+beta+1)*scipy.special.gamma(alpha+1)*scipy.special.gamma(beta+1)/scipy.special.gamma(alpha+beta+1)
  PL[0,:] = 1.0/np.sqrt(gamma0)
  if (N == 0):
    return PL
    
  gamma1 = (alpha+1)*(beta+1)/(alpha+beta+3)*gamma0
  PL[1,:] = ((alpha+beta+2)*LGL/2 + (alpha-beta)/2)/np.sqrt(gamma1)
  if (N == 1):
    return PL

  aold = 2/(2+alpha+beta)*np.sqrt((alpha+1)*(beta+1)/(alpha+beta+3));
  for i in range(1,N):
    h1 = 2*i+alpha+beta
    anew = 2/(h1+2)*np.sqrt( (i+1)*(i+1+alpha+beta)*(i+1+alpha)*(i+1+beta)/(h1+1)/(h1+3))
    bnew = - (alpha**2-beta**2)/h1/(h1+2)
    PL[i+1,:] = 1/anew*( -aold*PL[i-1,:] + (LGL-bnew)*PL[i,:])
    aold = anew

  return PL

"""
"""
def GaussLegendre(N):
    return np.polynomial.legendre.leggauss(N)

"""
"""
def VandermondeMatrices(N, phi, LGL):

  Vand = np.zeros((N,N))
  for i in range(0,N):
    for j in range(0,N):
      Vand[i,j] = np.polyval(phi[j],LGL[i])
      
  GradVand = np.zeros((N,N))
  for i in range(1,N):  
    GradVand[:,i] = np.sqrt(i*(i+1))*JacobiP(LGL,1.0,1.0,i,N)[i-1,:].T
    
  Dr = np.matmul(GradVand,np.linalg.inv(Vand))

  return Vand, np.linalg.inv(Vand), Dr
  
  
"""
"""
def Lift1D(N,Vand):

  Emat = np.zeros((N,2))
  Emat[ 0,0] = 1.0
  Emat[-1,1] = 1.0

  lift = np.matmul(Vand,np.matmul(Vand.T,Emat));
  
  return lift

"""
"""
def Jacobian(x, Dr):
  """
  Jacobian and its inverse
  """
  Jac = np.matmul(Dr,x)  
  return Jac, 1.0/Jac

"""
"""
def FaceToVertex(Ng):
  
  FtoV = np.zeros((2*Ng,Ng+1),dtype=np.int8)
  EtoV = np.zeros((Ng,2),dtype=np.int8)
  for i in range(0,Ng):
    EtoV[i,0] = int( i )
    EtoV[i,1] = int(i+1)
      
    FtoV[2*i  ,EtoV[i,0]] = 1
    FtoV[2*i+1,EtoV[i,1]] = 1

  return 


"""
"""
def Interpolation_Matrix(N, phi, xInt):

    Ninterp = xInt.shape[0]

    Interp = np.zeros((Ninterp,N))
    for i in range(0,Ninterp):
        for j in range(0,N):
            Interp[i,j] = np.polyval(phi[j],xInt[i])

    return Interp
