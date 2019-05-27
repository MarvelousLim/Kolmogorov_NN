import numpy as np, psi
gamma = 10
n = 3
limit = 15


class second_layer:
    #way to translate from float x to a number of constant sample is: sample[i] // (self.max / N)
    def __init__(self, N, n):
        """
        realization of second layer for GPU
        before start, please, initialize gamma, n and limit
        :param N: fraction number for piecewise constant function
        :param n: dimensionality of features
        """
        #adam params
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-8
        #not adam params
        self.arr = np.zeros((N, 2*n+1)) #param array
        self.max = sum([psi.lbd(p + 1, gamma, n) for p in range(n)])
        self.loss_memory = []
        self.eta_memory = []
        self.grad_memory = []
        self.N = N
        self.n = n
        
    def predict(self, X):
        """
        :param np.array X of size (size, 2n+1)
        :return np.array Y of size( size, 1) of predictions of model
        """
        result = []
        for x in X:
            prediction = 0
            for i in range(2*n+1):
                prediction += self.arr[int(x[i] // (self.max / N)), i]
            result.append(prediction)
        return np.array(result)
    
    def func_grad(self, X, Y):
        """calculates gradient for sum-squared error for
        :param X: np.array of size (size, 2n+1)
        :param Y: np.array of size( size, 1) of target values
        :return tuple (float sum squared error; gradient of error function np.array of size (N, 2*n+1) in point (X, Y))
        """
        Q = 0
        gradd = np.zeros((N, 2*n+1))
        for x, y in zip(X, Y):
            #pairs of required arr coordinates
            iterat = [(int(x[i] // (self.max / N)), i) for i in range(2*n+1)]
            prediction = 0
            for j, i in iterat:
                prediction += self.arr[j, i]
            delta = prediction - y
            #in a meantime I precalculate new_step and, if all right, Ill use it to make new step
            for j, i in iterat:
                gradd[j, i] += delta    
            Q += delta * delta
        return Q / len(X), gradd / len(X)


#    def fit(self, X, Y, eta=0.01, n_steps=10):
#        F_old, grad = self.func_grad(X, Y)
#        for p in range(n_steps):
#            #saves
#            arr_old = self.arr.copy()
#            grad_old = grad.copy()
#            #pre_step
#            self.arr -= eta * grad
#            #recalculation
#            F, grad = self.func_grad(X, Y)
#            #step or no step
#            if F < F_old:
#                F_old = F
#                eta *= 1.25
#            else:
#                self.arr = arr_old
#                grad = grad_old
#                eta /= 1.25
#            self.loss_memory.append(F_old)
#            self.eta_memory.append(eta)
#            self.grad_memory.append(np.sum(grad * grad) 
#/ (N * (2*n+1)))

    def fit(self, X, Y, alpha=0.1, n_steps=10):
        """
        learning with adam
        :param X: np.array of size (size, 2n+1) - features
        :param Y: np.array of size (size, 1) of target values
        :param alpha: float, learning rate
        :param n_steps: int, number of steps
        """
        self.alpha = alpha
        #initialize tmp parameters
        m = np.zeros_like(self.theta)
        v = np.zeros_like(self.theta)
        t = 0
        for p in range(n_steps):
            t += 1
            F, grad = self.func_grad(X, Y) #grad is equivalent to g
            m = self.beta_1 * m + (1 - self.beta_1) * grad
            v = self.beta_2 * v + (1 - self.beta_2) * grad * grad
            m_hat = m / (1 - pow(self.beta_1, t))
            v_hat = v / (1 - pow(self.beta_2, t))
                        
            self.theta -= self.alpha * m_hat / (self.epsilon + np.power(v_hat, 0.5))
            
            self.loss_memory.append(F)
            self.grad_memory.append(np.sum(grad * grad) / (N * (2*n+1)))


    def set_params(self, arr):
        """allows one to set parameters of piesewise constant fucntions on your own
        - arr - np.array of size (N, 2n+1)"""
        self.arr = arr
    def get_params(self):
        """allows one to get parameters of piesewise constant fucntions
        :returns
        - arr - np.array of size (N, 2n+1)"""
        return self.arr
    def get_memory(self):
        """allows one to get values of error function, obtained during learning
        :returns
        - loss_memory - np.array of size (n_steps, 1)"""
        return self.loss_memory