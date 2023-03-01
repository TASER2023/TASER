import numpy as np
import random

class POS2:
    # this is for 2D
    def __init__(self, pN, dim1, dim2, X):
        self.w = 1
        self.c1 = 2
        self.c2 = 2
        self.pN = pN  # the number of particles
        self.dim1 = dim1 
        self.dim2 = dim2 
        self.X = X  # particles
        # self.V = np.random.rand(self.pN, self.dim1, self.dim2)
        self.V = np.random.uniform(-0.2, 0.2, (self.pN, self.dim1, self.dim2))  # the velocity of all particles
        self.pbest = X  # personal best position
        # self.p_fit = np.zeros(self.pN)
        self.p_fit = np.ones(self.pN)*100000 # personal best value
        self.gbest = np.zeros((1, self.dim1, self.dim2))  # global best position
        self.best=-1  # the index of the best particle
        # self.fit = -1e10
        self.fit = 100000  # global best value    

    # update once
    def update(self, loss):    
        flag=False  # whether find the best particle
        for i in range(self.pN):
            temp=loss[i]
            if temp<self.p_fit[i]:
                self.p_fit[i]=temp
                self.pbest[i]=self.X[i].copy()
                if temp<self.fit:
                    self.fit=temp
                    self.gbest=self.X[i].copy()
                    # print(f"gbest={self.gbest}, fit={self.fit}")
                    self.best=i
                    flag=True
        
        for i in range(self.pN):
            r1=np.random.rand()
            r2=np.random.rand()
            tempV=self.V[i].copy()
            tempX=self.V[i].copy()
            self.V[i] = self.w * self.V[i] + self.c1 * r1 * (self.pbest[i] - self.X[i]) + \
                        self.c2 * r2 * (self.gbest - self.X[i])
            self.X[i] = self.X[i] + self.V[i]
            if np.sum((np.abs(self.X[i])>0.4)) >1:
                self.V[i]=tempV
                self.X[i]=tempX
        # print(self.gbest)
        return self.X, self.gbest, self.best, flag
