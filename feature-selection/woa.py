import numpy as np
from numpy.random import rand
from sklearn.neighbors import KNeighborsClassifier

class WhaleOptimization():
    def init_position(self, lb, ub, N, dim):
        X = np.zeros([N, dim], dtype='float')
        for i in range(N):
            for d in range(dim):
                X[i,d] = lb[0,d] + (ub[0,d] - lb[0,d]) * rand()        
        
        return X

    def binary_conversion(self, X, thres, N, dim):
        Xbin = np.zeros([N, dim], dtype='int')
        for i in range(N):
            for d in range(dim):
                if X[i,d] > thres:
                    Xbin[i,d] = 1
                else:
                    Xbin[i,d] = 0
        
        return Xbin
    
    def boundary(self, x, lb, ub):
        if x < lb:
            x = lb
        if x > ub:
            x = ub
        
        return x

    def error_rate(self, xtrain, ytrain, x, opts):
        # parameters
        k     = opts['k']
        fold  = opts['fold']
        xt    = fold['xt']
        yt    = fold['yt']
        xv    = fold['xv']
        yv    = fold['yv']
        
        # Number of instances
        num_train = np.size(xt, 0)
        num_valid = np.size(xv, 0)
        # Define selected features
        xtrain  = xt[:, x == 1]
        ytrain  = yt.reshape(num_train)  # Solve bug
        xvalid  = xv[:, x == 1]
        yvalid  = yv.reshape(num_valid)  # Solve bug   
        # Training
        mdl     = KNeighborsClassifier(n_neighbors = k)
        mdl.fit(xtrain, ytrain)
        # Prediction
        ypred   = mdl.predict(xvalid)
        acc     = np.sum(yvalid == ypred) / num_valid
        error   = 1 - acc
        
        return error

    # Error rate & Feature size
    def Fun(self, xtrain, ytrain, x, opts):
        # Parameters
        alpha    = 0.99
        beta     = 1 - alpha
        # Original feature size
        max_feat = len(x)
        # Number of selected features
        num_feat = np.sum(x == 1)
        # Solve if no feature selected
        if num_feat == 0:
            cost  = 1
        else:
            # Get error rate
            error = self.error_rate(xtrain, ytrain, x, opts)
            # Objective function
            cost  = alpha * error + beta * (num_feat / max_feat)
            
        return cost

    def jfs(self, xtrain, ytrain, opts):
        # Parameters
        ub    = 1
        lb    = 0
        thres = 0.5
        b     = 1       # constant
        
        N        = opts['N']
        max_iter = opts['T']
        if 'b' in opts:
            b    = opts['b']
        
        # Dimension
        dim = np.size(xtrain, 1)
        if np.size(lb) == 1:
            ub = ub * np.ones([1, dim], dtype='float')
            lb = lb * np.ones([1, dim], dtype='float')
            
        # Initialize position 
        X    = self.init_position(lb, ub, N, dim)
        
        # Binary conversion
        Xbin = self.binary_conversion(X, thres, N, dim)
        
        # Fitness at first iteration
        fit  = np.zeros([N, 1], dtype='float')
        Xgb  = np.zeros([1, dim], dtype='float')
        fitG = float('inf')
        
        for i in range(N):
            fit[i,0] = self.Fun(xtrain, ytrain, Xbin[i,:], opts)
            if fit[i,0] < fitG:
                Xgb[0,:] = X[i,:]
                fitG     = fit[i,0]
            
        # Pre
        curve = np.zeros([1, max_iter], dtype='float') 
        t     = 0
        
        curve[0,t] = fitG.copy()
        print("Generation:", t + 1)
        print("Best (WOA):", curve[0,t])
        t += 1

        while t < max_iter:
            # Define a, linearly decreases from 2 to 0 
            a = 2 - t * (2 / max_iter)
            
            for i in range(N):
                # Parameter A (2.3)
                A = 2 * a * rand() - a
                # Paramater C (2.4)
                C = 2 * rand()
                # Parameter p, random number in [0,1]
                p = rand()
                # Parameter l, random number in [-1,1]
                l = -1 + 2 * rand()  
                # Whale position update (2.6)
                if p  < 0.5:
                    # {1} Encircling prey
                    if abs(A) < 1:
                        for d in range(dim):
                            # Compute D (2.1)
                            Dx     = abs(C * Xgb[0,d] - X[i,d])
                            # Position update (2.2)
                            X[i,d] = Xgb[0,d] - A * Dx
                            # Boundary
                            X[i,d] = self.boundary(X[i,d], lb[0,d], ub[0,d])
                    
                    # {2} Search for prey
                    elif abs(A) >= 1:
                        for d in range(dim):
                            # Select a random whale
                            k      = np.random.randint(low = 0, high = N)
                            # Compute D (2.7)
                            Dx     = abs(C * X[k,d] - X[i,d])
                            # Position update (2.8)
                            X[i,d] = X[k,d] - A * Dx
                            # Boundary
                            X[i,d] = self.boundary(X[i,d], lb[0,d], ub[0,d])
                
                # {3} Bubble-net attacking 
                elif p >= 0.5:
                    for d in range(dim):
                        # Distance of whale to prey
                        dist   = abs(Xgb[0,d] - X[i,d])
                        # Position update (2.5)
                        X[i,d] = dist * np.exp(b * l) * np.cos(2 * np.pi * l) + Xgb[0,d] 
                        # Boundary
                        X[i,d] = self.boundary(X[i,d], lb[0,d], ub[0,d])
            
            # Binary conversion
            Xbin = self.binary_conversion(X, thres, N, dim)
            
            # Fitness
            for i in range(N):
                fit[i,0] = self.Fun(xtrain, ytrain, Xbin[i,:], opts)
                if fit[i,0] < fitG:
                    Xgb[0,:] = X[i,:]
                    fitG     = fit[i,0]
            
            # Store result
            curve[0,t] = fitG.copy()
            print("Generation:", t + 1)
            print("Best (WOA):", curve[0,t])
            t += 1            

        # Best feature subset
        Gbin       = self.binary_conversion(Xgb, thres, 1, dim) 
        Gbin       = Gbin.reshape(dim)    
        pos        = np.asarray(range(0, dim))    
        sel_index  = pos[Gbin == 1]
        num_feat   = len(sel_index)
        # Create dictionary
        woa_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}
        
        return woa_data 