import numpy as np
import random

class Memoria():

    def __init__(self,cantidad):
        self.cantidad = cantidad
        self.memoria = np.array([[[[0,0,0,0]],0,0,[[0,0,0,0]],[[0,0]]]])

        for i in range(cantidad):
            m = self.memoria
            a = np.array([[[[0,0,0,0]],0,0,[[0,0,0,0]],[[0,0]]]])
            self.memoria = np.vstack([m,a])

    def agregar(self,estado,accion,reward,estado_,target):

        self.memoria = np.vstack([[estado,accion,reward,estado_,target],self.memoria])
        self.memoria = self.memoria[:self.cantidad]
        return self.memoria

    def leer(self,batch_size):
        lista = random.sample(range(self.cantidad),batch_size)
        batch = np.zeros((1,5))
        for i in lista:
            item = self.memoria[i]
            batch = np.vstack([item,batch])


        return batch

class RL():

    def __init__(self):
        self.memoria = np.zeros((100,5))

    def discretizar(nmin, nmax, num, steps):
        t = []
        c = 0
        for i in range(0,steps):
            t.append(nmin)
        stepsize = (nmax - nmin) / steps

        for i in range(1, steps):
            t[i] = t[i - 1] + stepsize

        for i in range(1,steps):
            if num >= t[i-1] and num < t[i]:
                c = i - steps//2
            if num >= nmax:
                c = steps//2
            if num < nmin:
                c = steps//-2
        return c

