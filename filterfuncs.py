import numpy as np

"""
"""
def minmod(a1,a2,a3):

    Ng = a1.shape[0]

    s = (1.0/3.0)*(np.sign(a1)+np.sign(a2)+np.sign(a3))
    flagS = np.isclose(np.absolute(s),1.0)

    minimum = s*np.minimum(np.abs(a1),np.abs(a2),np.abs(a3))
    minmodFunction = np.zeros((Ng))
    minmodFunction[np.where(flagS)] = minimum[np.where(flagS)]
        
    return minmodFunction
    
def limiter1(u,umin,umax,Vand,invVand,Dr,x,xm,dx):

    #TODO truncate matrices to avoid unnecessary products

    uLimited = np.zeros((u.shape))
    uhat = np.zeros((u.shape))
    ubar = np.zeros((u.shape))
    uave = np.zeros((u.shape[1],3))

    uhat = np.matmul(invVand,u)
    uhat[2:,:] = 0.0
    utilde = np.matmul(Vand,uhat)
    
    ubar = np.matmul(invVand,u)
    ubar[1:,:] = 0.0
    uave[ :,1] = np.matmul(Vand,ubar)[0,:]
    uave[0:-1,2] = uave[  1:,1]
    uave[  -1,2] = uave[  -1,1]
    uave[  1:,0] = uave[0:-1,1]
    uave[   0,0] = uave[   0,1]
    
    a1 = 2.0/dx*np.matmul(Dr,utilde)[0,:]
    a2 = 1.0/dx*(uave[:,1] - uave[:,0])
    a3 = 1.0/dx*(uave[:,2] - uave[:,1])

    uLimited = uave[:,1] + (x-xm)*minmod(a1,a2,a3)
    
    return uLimited

def limiter2(u,umin,umax,Vand,invVand,Dr,x,xm,dx):

    #TODO truncate matrices to avoid unnecessary products

    uLimited = np.zeros((u.shape))
    uhat = np.zeros((u.shape))
    ubar = np.zeros((u.shape))
    uave = np.zeros((u.shape[1],3))
        
    ue0 = u[ 0,:]
    ueN = u[-1,:]

    uhat = np.matmul(invVand,u)
    uhat[1:,:] = 0.0
    
    uave[ :,1] = np.matmul(Vand,uhat)[0,:]
    uave[0:-1,2] = uave[  1:,1]
    uave[  -1,2] = uave[  -1,1]
    uave[  1:,0] = uave[0:-1,1]
    uave[   0,0] = uave[   0,1]
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #minmod input    
    a1 = uave[:,1] - ue0
    a2 = uave[:,1] - uave[:,0]
    a3 = uave[:,2] - uave[:,1]    
    
    ve1 = uave[:,1] - minmod(0.5*a1,a2,a3) #- ue0
    flag1 = np.where(np.abs(ve1) >= 1e-7, True, False)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #minmod input    
    a1 = ueN - uave[:,1]
    a2 = uave[:,1] - uave[:,0]
    a3 = uave[:,2] - uave[:,1]
    
    ve2 = uave[:,1] + minmod(0.5*a1,a2,a3) #- ueN
    flag2 = np.where(np.abs(ve2) >= 1e-7, True, False)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    flagE = np.logical_or(flag1,flag2)

    a1 = ueN - ue0
    a2 = uave[:,1] - uave[:,0]
    a3 = uave[:,2] - uave[:,1]
    
    #slope limiting
    uLimited[:,:] = np.where(flagE,
                             uave[:,1] + (x-xm)*minmod(a1,a2,a3)/dx,
                             u[:,:])
    
    return uLimited

def limiterN(u,umin,umax,Vand,invVand,Dr,x,xm,dx):

    #TODO truncate matrices to avoid unnecessary products

    uLimited = np.zeros((u.shape))
    uhat = np.zeros((u.shape))
    ubar = np.zeros((u.shape))
    uave = np.zeros((u.shape[1],3))
        
    ue0 = u[ 0,:]
    ueN = u[-1,:]

    uhat = np.matmul(invVand,u)
    uhat[1:,:] = 0.0
    
    uave[ :,1] = np.matmul(Vand,uhat)[0,:]
    uave[0:-1,2] = uave[  1:,1]
    uave[  -1,2] = uave[  -1,1]
    uave[  1:,0] = uave[0:-1,1]
    uave[   0,0] = uave[   0,1]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #minmod input    
    a1 = uave[:,1] - ue0
    a2 = uave[:,1] - uave[:,0]
    a3 = uave[:,2] - uave[:,1]  
#    a1 = uave[:,1] - ue0
#    a2 = ue0 - uave[:,0]
#    a3 = uave[:,2] - ueN       
    
    ve1 = uave[:,1] - minmod(a1,a2,a3) - ue0
    flag1 = np.where(np.abs(ve1) >= 1e-7, True, False)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #minmod input    
    a1 = ueN - uave[:,1]
    a2 = uave[:,1] - uave[:,0]
    a3 = uave[:,2] - uave[:,1]
#    a1 = ueN - uave[:,1]
#    a2 = ue0 - uave[:,0]
#    a3 = uave[:,2] - ueN    
    
    ve2 = uave[:,1] + minmod(a1,a2,a3) - ueN
    flag2 = np.where(np.abs(ve2) >= 1e-7, True, False)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    flagE = np.logical_or(flag1,flag2)
    
    #slope limiting
    ubar = np.matmul(invVand,u)    
    ubar[2:,flagE] = 0.0
    utilde = np.matmul(Vand,ubar)

    #minmod input
    a1 = 1.0/dx*np.matmul(Dr,utilde)[0,:]
    a2 = 1.0/dx*(uave[:,1] - uave[:,0])
    a3 = 1.0/dx*(uave[:,2] - uave[:,1])
#    a1 = 1.0/dx*np.matmul(Dr,utilde)[0,:]
#    a2 = 1.0/dx*(ue0 - uave[:,0])
#    a3 = 1.0/dx*(uave[:,2] - ueN)     

    uLimited[:,:] = np.where(flagE,
                             uave[:,1] + (x-xm)*minmod(a1,a2,a3),
                             u[:,:])
    
    return uLimited



def ShuLimiter(u,uMaxVal,uMinVal,Vand,invVand,Dr,x,xm,dx):

    #TODO truncate matrices to avoid unnecessary products

    uLimited = np.zeros((u.shape))
    umax = np.zeros((u.shape[1]))
    umin = np.zeros((u.shape[1]))
    uave = np.zeros((u.shape[1],3))
        
    uhat = np.matmul(invVand,u)
    uhat[1:,:] = 0.0
    
    uave[ :,1] = np.matmul(Vand,uhat)[0,:]
    uave[0:-1,2] = uave[  1:,1]
    uave[  -1,2] = uave[   0,1]
    uave[  1:,0] = uave[0:-1,1]
    uave[   0,0] = uave[  -1,1]

    umin[:] = np.amin(u,axis=0)
    umax[:] = np.amax(u,axis=0)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #minmod input    
    a1 = 1.0
    
    a2a = (uave[:,1] - uave[:,0])/(umax - uave[:,1] + 1e-12)
    a2b = (uave[:,2] - uave[:,1])/(umax - uave[:,1] + 1e-12)
    a2 = np.minimum(a2a,a2b)
    
    a3a = (uave[:,1] - uave[:,0])/(umin - uave[:,1] + 1e-12)
    a3b = (uave[:,2] - uave[:,1])/(umin - uave[:,1] + 1e-12)    
    a3 = np.minimum(a3a,a3b)
    
    theta = np.minimum(np.abs(a1),np.abs(a2),np.abs(a3))

    uLimited[:,:] = uave[:,1]  + theta*(u[:,:] - uave[:,1])                          
    
    return uLimited

def ShuLimiter_bound(u,uMaxVal,uMinVal,Vand,invVand,Dr,x,xm,dx):

    uLimited = np.zeros((u.shape))
    umax = np.zeros((u.shape[1]))
    umin = np.zeros((u.shape[1]))
    uave = np.zeros((u.shape[1],3))

    umin[:] = np.amin(u,axis=0)
    umax[:] = np.amax(u,axis=0)
            
    uhat = np.matmul(invVand,u)
    uhat[1:,:] = 0.0
    
    uave[   :,1] = np.matmul(Vand,uhat)[0,:]
    uave[  1:,0] = u[-1,0:-1]
    uave[   0,0] = u[-1,-1]
    uave[0:-1,2] = u[ 0,1:]
    uave[  -1,2] = u[ 0, 0]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #minmod input    
    a1 = 1.0
    a2a = (uave[:,1] - uave[:,0])/(umax - uave[:,1] + 1e-12)
    a2b = (uave[:,2] - uave[:,1])/(umax - uave[:,1] + 1e-12)
    a2 = np.minimum(np.abs(a2a),np.abs(a2b))
    a3a = (uave[:,1] - uave[:,0])/(umin - uave[:,1] + 1e-12)
    a3b = (uave[:,2] - uave[:,1])/(umin - uave[:,1] + 1e-12)    
    a3 = np.minimum(np.abs(a3a),np.abs(a3b))
    
    theta = np.minimum(a1,a2,a3)

    uLimited[:,:] = uave[:,1]  + theta*(u[:,:] - uave[:,1])                          

    #_theta = np.vstack((theta,theta))    
    return uLimited#, uave[:,0], uave[:,1], uave[:,2], _theta
    
################################################################################

"""
"""
def filter1D(N,Nc,s,Vand,invVand):

    filterdiag = np.eye((N))
    alpha = 2.0
    
#    print(Nc,N)

    if (Nc == 0):
      #print(filterdiag)
      return filterdiag

    if (Nc >= N):
      #print(filterdiag)
      return filterdiag
    
    if (Nc >= N-1):
      filterdiag[-1,-1] = 0.0
      #print(filterdiag)
      return filterdiag
    
    for i in range(Nc,N):
        print(i)
        filterdiag[i,i] = np.exp(-alpha*((i-Nc)/((N-1)-Nc))**s)
    print(filterdiag)
        
    return np.matmul(Vand,np.matmul(filterdiag,invVand))
