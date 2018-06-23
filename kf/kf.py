import numpy as np

class KF():
    def __init__(self, A, B, H, R, Q):
        self.A = A
        self.B = B
#        self.H = H
        self.H = np.atleast_2d(H)
        self.Q = Q
        self.P = np.eye(A.shape[0]) * 1000.
        self.x = np.zeros(A.shape[0])
        self.log_x = []
        self.xi = np.zeros(np.asarray(self.P.shape) + 1)
        if np.isscalar(R):
            self.Rinv = 1/R
        else:
            self.Rinv = np.linalg.inv(R)

        
    def predict(self, u=None):
        xminus = self.A.dot(self.x) 
        if u is not None:
            xminus += self.B.dot(u)
        Pminus = self.A.dot(self.P).dot(self.A.T) + self.Q
        xi_vector = np.r_[xminus[np.newaxis], np.eye(self.x.shape[0])]
        self.xi = xi_vector.dot(np.linalg.inv(Pminus)).dot(xi_vector.T)
        self.x = xminus   # temporary
        self.P = Pminus   # temporary
        
    def update(self, y, Rinv=None):
        if Rinv is None:
            Rinv = self.Rinv
        y = np.atleast_2d(y).reshape(self.H.shape[0], -1)
        T_vector = np.concatenate((y, self.H), axis=1) 
        T = T_vector.T.dot(Rinv).dot(T_vector)
        self.xi += T
        self.P = np.linalg.inv(self.xi[1:,1:])
        self.x = self.P.dot(self.xi[1:,0])
        
    def log(self):
        self.log_x.append(self.x.copy())


#--------------
if __name__ == '__main__':
    from scipy.stats import multivariate_normal as mvn
    from scipy.stats import norm
    import matplotlib.pylab as plt
    
    test = 'bivariate'
    
    if test == 'univariate':
        ndat = 100                               ### počet dat
        h0 = 0                                   ### počáteční výška [m]
        v0 = 520                                 ### počáteční rychlost [m/s]
        g = 9.81                                 ### grav. zrychlení [m/s^2]
        dt = 1                                   # čas. krok v sekundách
        A = np.array([[1, dt], [0, 1]])          # matice A
        B = np.array([-.5*dt, -dt])              # matice B
        u = g                                    # pro forma, g=u
        var_wht = 10                             ### variance šumu na výšce w_ht
        var_wvt = 10                             ### variance šumu na rychlosti w_vt
        var_y = 90000                            ### variance šumu měření v_t
        
        x = np.zeros((2, ndat))
        x[:,0] = [h0, v0]
        y = np.zeros(ndat)
        
        for t in range(1, ndat):
            x[:,t] = A.dot(x[:,t-1]) + B.dot(u)
            x[:,t] += mvn.rvs(cov=np.diag([var_wht, var_wvt]))
            y[t] = x[0,t] + norm.rvs(scale=np.sqrt(var_y))
    
    #---------
        Q = np.diag([var_wht, var_wvt])           # Toto už jsme v simulaci také udělali
        R = var_y
        H = np.array([1, 0])
        
        filtr = KF(A=A, B=B, H=H, R=R, Q=Q)       # Instance Kalmanova filtru
        for yt in y:
            filtr.predict(u)
            filtr.update(yt)
            filtr.log()
    #        input('asdf')
        log_x = np.array(filtr.log_x).T
    #----------        
        plt.figure(figsize=(14,4))
        plt.subplot(221)
        plt.plot(x[0], label='Skutečná h')
        plt.plot(y, '+r', label='Měřená h')
        plt.legend()
        plt.xlabel('t')
        plt.subplot(222)
        plt.plot(x[1], label='skutečná v')
        plt.plot(np.abs(x[1]), '--b', label='abs. hodnota v')
        plt.legend()
        plt.xlabel('t')
        plt.subplot(223)
        plt.plot(x[0])
        plt.plot(log_x[0])
        plt.subplot(224)
        plt.plot(x[1])
        plt.plot(log_x[1])
        plt.show()
    #==================================
    
    elif test == 'bivariate':
        q = 5.
        dt = .1
        r = .2
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1,  0],
                      [0, 0, 0,  1]])
        Q = q * np.array([[dt**3/3, 0      , dt**2/2, 0      ],
                          [0,       dt**3/3, 0,       dt**2/2],
                          [dt**2/2, 0,       dt,      0      ],
                          [0,       dt**2/2, 0,       dt     ]])
        H = np.array([[1., 0, 0, 0],
                      [0., 1, 0, 0]])
        R = r**2 * np.eye(2)
        m0 = np.array([0., 0., 1., -1.]).reshape(4,1)
        P0 = np.eye(4)
        
        #--- Simulace
        np.random.seed(1234567890)
        steps = 100
        X = np.zeros(shape=(A.shape[0], steps))
        Y = np.zeros(shape=(H.shape[0], steps))
        Y_clear = np.zeros(shape=(H.shape[0], steps))
        x = m0;
        for t in range(steps):
            q = np.linalg.cholesky(Q).dot(np.random.normal(size=(A.shape[0], 1)))
            x = A.dot(x) + q
            y_clear = np.dot(H, x)
            y = y_clear + r * np.random.normal(size=(2,1))
            X[:,t] = x.flatten()
            Y[:,t] = y.flatten()
            Y_clear[:,t] = y_clear.flatten()
        
        #---- Odhady
        kf = KF(A=A, B=None, H=H, R=R, Q=Q)
        for y in Y.T:
            kf.predict()
            kf.update(y)
            kf.log()
#            input('asdf')
            
            log_x = np.array(kf.log_x).T
            
        #--- vysledky
        plt.figure(figsize=(10,10))
        plt.plot(X[0], X[1], '-', label='Trajectory')
        plt.plot(Y[0], Y[1], '.', label='Observations')
        plt.plot(log_x[0], log_x[1], '-', color='red', label='Est')
        plt.grid(True)
        plt.legend()