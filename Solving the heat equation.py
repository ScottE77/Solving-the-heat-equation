import math
import numpy as np
import matplotlib.pyplot as plt
import statistics
    

def analyticalsolution(N=5,C=1/3,t=0.25,K=1,L=1):#not finished
    deltax = 1/(N-1)
    deltat = (C*((deltax)**2))/K
    theta = np.zeros([int(t//deltat)+1,N])
    for m in range(int(t//deltat)+1):
        for i in range(N):
            theta[m][i] = math.sin((math.pi*deltax*i)/L)*math.exp(-K*((math.pi/L)**2)*(deltat*m))#An = 1 when n is 1 and 0 when n isn't 1
    return theta#last value is machine error


def simplefinitedifference(N=5,C=1/3,t=0.25,K=1,L=1):
    deltax = 1/(N-1)
    deltat = (C*((deltax)**2))/K
    theta = np.zeros([int(t//deltat)+1,N])
    for i in range(N-2):
        theta[0][i+1] = (math.sin((math.pi)*(i+1)*deltax))
    for m in range(int(t//deltat)):
        for i in range(N-2):
            theta[m+1][i+1] = theta[m][i+1] + C*(theta[m][i+2]-2*(theta[m][i+1])+theta[m][i])
    #plt.plot(simplefinitedifference(N,C,t,K,L))
    return theta

def plots(N=5,C=1/3,t=0.25,K=1,L=1):
    deltax = 1/(N-1)
    allx = [0]
    for i in range(N-1):
        allx.append(allx[-1]+deltax)
    np.array(allx)
    analytical = analyticalsolution(N,C,t,K,L)[-1]
    simple = simplefinitedifference(N,C,t,K,L)[-1]
    plt.figure(1)
    plt.subplot(211)
    plt.plot(allx,analytical)
    #plt.figure(2)
    #plt.subplot(212)
    for i in range(len(simple)):
        simple[i] = abs(simple[i])
    #plt.plot(allx,simple)
    plt.xlabel('x')
    plt.ylabel('Temperatuture')
    error = []
    for i in range(N-2):
        error.append(abs(analytical[i+1]-simple[i+1]))
    return error

def error(N=5,C=1/3,t=0.25):
    error = 0
    allx = [0]
    deltax = 1/(N-1)
    for i in range(N-1):
        allx.append(allx[-1]+deltax)
    simple = simplefinitedifference(N,C,t)[-1]
    analytical = analyticalsolution(N,C,t)[-1]
    A,B = [],[]
    for i in range(N):
        A.append([allx[i],simple[i]])
        B.append([allx[i],analytical[i]])
    for i in range(N-1):
        mid = [(A[i][0]+B[i+1][0])/2,(A[i][1]+B[i+1][1])/2]
        base = math.sqrt((A[i][0]-B[i+1][0])**2+(A[i][1]-B[i+1][1])**2)
        h1 = math.sqrt(((mid[0]-B[i][0])**2+(mid[1]-B[i][1])**2))
        h2 = math.sqrt(((mid[0]-A[i+1][0])**2+(mid[1]-A[i+1][1])**2))
        a1 = (h1*base)/2
        a2 = (h2*base)/2
        error += a1+a2
    return error

def errorplotsN(n,C=1/3,t=0.25):
    allN = []
    for i in range(n):
        allN.append(i+3)#start from 3 because 1 and 2 are useless
    errorN = []
    for i in range(n):
        errorN.append(error(allN[i],C,t))
    plt.plot(allN,errorN)
    plt.xlabel('N')
    plt.ylabel('Error')
        
def errorplotsC(n,maxC=1/2,N=5,t=0.25):
    deltaC = maxC/(n-1)
    allC = []
    for i in range(n-1):
        allC.append((i+1)*deltaC)
    errorC = []
    for i in range(n-1):
        errorC.append(error(N,allC[i],t))
    plt.plot(allC,errorC)
    plt.xlabel('C')
    plt.ylabel('Error')
    
def meanerrorC(n,maxC=1/2,N=5,t=0.25):
    deltaC = maxC/(n-1)
    allC = []
    for i in range(n-1):
        allC.append((i+1)*deltaC)
    errorC = []
    for i in range(n-1):
        errorC.append(error(N,allC[i],t))
    return (statistics.mean(errorC))
     
