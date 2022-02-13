class PolynomialRegression() :
    
    def __init__(self, m=1) :
        """
        Ordinary least squares regression.
        
        Attributes
        --------------------
            coef_   -- numpy array of shape (d,)
                       estimated coefficients for the linear regression problem
            m_      -- integer
                       order for polynomial regression
        """
        self.coef_ = None
        self.m_ = m
    
    
    def generate_polynomial_features(self, X) :
        """
        Maps X to an mth degree feature vector e.g. [1, X, X^2, ..., X^m].
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,1), features
        
        Returns
        --------------------
            Phi     -- numpy array of shape (n,(m+1)), mapped features
        """
        
        n,d = X.shape
        
        ### ========== TODO : START ========== ###
        # part b: modify to create matrix for simple linear model
        intercepts = np.ones((n, 1))
        m = self.m_
        if (m == 0):
            return intercepts
        Phi = X
        Phi = np.concatenate((intercepts, Phi), axis=1)
        # part g: modify to create matrix for polynomial model
        for i in (range(m-1)):
            Phi = np.concatenate((Phi, X ** (i + 2)), axis=1)

        ### ========== TODO : END ========== ###
        return Phi
    
    
    def fit_GD(self, X, y, eta=None,
                eps=0, tmax=10000, verbose=False) :
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using least squares batch gradient descent.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
            eta     -- float, step size
            eps     -- float, convergence criterion
            tmax    -- integer, maximum number of iterations
            verbose -- boolean, for debugging purposes
        
        Returns
        --------------------
            self    -- an instance of self
        """
        
        if verbose :
            plt.subplot(1, 2, 2)
            plt.xlabel('iteration')
            plt.ylabel(r'$J(\theta)$')
            plt.ion()
            plt.show()
        
        X = self.generate_polynomial_features(X) # map features
        n,d = X.shape
        eta_input = eta
        self.coef_ = np.zeros(d)                 # coefficients
        err_list  = np.zeros((tmax,1))           # errors per iteration
        
        # GD loop
        for t in range(tmax) :
            ### ========== TODO : START ========== ###
            # part f: update step size
            # change the default eta in the function signature to 'eta=None'
            # and update the line below to your learning rate function
            if eta_input is None :
                eta = 1/float(1 + t) # change this line
            else :
                eta = eta_input
            ### ========== TODO : END ========== ###
            
            ### ========== TODO : START ========== ###
            # part d: update theta (self.coef_) using one step of GD
            # hint: you can write simultaneously update all theta using vector math
            
            self.coef_ -=  2 * eta * np.dot(np.dot(self.coef_, np.transpose(X)) - y, X)
                
            # track error
            # hint: you cannot use self.predict(...) to make the predictions
            y_pred = np.dot(self.coef_, np.transpose(X)) # change this line
            err_list[t] = np.sum(np.power(y - y_pred, 2)) / float(n)                
            ### ========== TODO : END ========== ###
            
            # stop?
            if t > 0 and abs(err_list[t] - err_list[t-1]) <= eps :
                break
            
            # debugging
            if verbose :
                x = np.reshape(X[:,1], (n,1))
                cost = self.cost(x,y)
                plt.subplot(1, 2, 1)
                plt.cla()
                plot_data(x, y)
                self.plot_regression()
                plt.subplot(1, 2, 2)
                plt.plot([t+1], [cost], 'bo')
                plt.suptitle('iteration: %d, cost: %f' % (t+1, cost))
                plt.draw()
                plt.pause(0.05) # pause for 0.05 sec
        
        print('number of iterations: %d' % (t+1))
        
        return self
    
    
    def fit(self, X, y) :
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using the closed form solution.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
            
        Returns
        --------------------        
            self    -- an instance of self
        """
        
        X = self.generate_polynomial_features(X) # map features
        
        ### ========== TODO : START ========== ###
        # part e: implement closed-form solution
        # hint: use np.dot(...) and np.linalg.pinv(...)
        #       be sure to update self.coef_ with your solution
        self.coef_ = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(X), X)), np.transpose(X)), y)
        
        ### ========== TODO : END ========== ###
    
    
    def predict(self, X) :
        """
        Predict output for X.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
        
        Returns
        --------------------
            y       -- numpy array of shape (n,), predictions
        """
        if self.coef_ is None :
            raise Exception("Model not initialized. Perform a fit first.")
        
        X = self.generate_polynomial_features(X) # map features
        
        ### ========== TODO : START ========== ###
        # part c: predict y
        y = np.dot(self.coef_, np.transpose(X))
        ### ========== TODO : END ========== ###
        
        return y
    
    
    def cost(self, X, y) :
        """
        Calculates the objective function.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        
        Returns
        --------------------
            cost    -- float, objective J(theta)
        """
        ### ========== TODO : START ========== ###
        # part d: compute J(theta)
        n,d = X.shape
        X = self.generate_polynomial_features(X)
        cost = np.sum((np.dot(self.coef_, np.transpose(X)) - y) ** 2)
        ### ========== TODO : END ========== ###
        return cost
    
    
    def rms_error(self, X, y) :
        """
        Calculates the root mean square error.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        
        Returns
        --------------------
            error   -- float, RMSE
        """
        ### ========== TODO : START ========== ###
        # part h: compute RMSE
        n,d = X.shape
        error = np.sqrt(self.cost(X, y) / n)
        ### ========== TODO : END ========== ###
        return error
    
    
    def plot_regression(self, xmin=0, xmax=1, n=50, **kwargs) :
        """Plot regression line."""
        if 'color' not in kwargs :
            kwargs['color'] = 'r'
        if 'linestyle' not in kwargs :
            kwargs['linestyle'] = '-'
        
        X = np.reshape(np.linspace(0,1,n), (n,1))
        y = self.predict(X)
        plot_data(X, y, **kwargs)
        plt.show()

######################################################################
# main
######################################################################

def main():
    # load data
    train_data = load_data(train_path)
    test_data = load_data(test_path)
    
    
    ### ========== TODO : START ========== ###
    # part a: main code for visualizations
    print('Visualizing data...')
    plot_data(train_data.X, train_data.y)
    plot_data(test_data.X, test_data.y)
    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # parts b-f: main code for linear regression
    print('Investigating linear regression...')
    lin_reg = PolynomialRegression()
    lin_reg.coef_ = np.zeros(2)
    print("Cost:", lin_reg.cost(train_data.X, train_data.y))
    cp0 = time.time()
    lin_reg.fit_GD(train_data.X, train_data.y, eta=10**-6)
    cp1 = time.time()
    print("eta = 10^-6:", lin_reg.coef_)
    print("Time eta = 10^-6:", cp1 - cp0)
    print("Cost:", lin_reg.cost(train_data.X, train_data.y))
    cp0 = time.time()
    lin_reg.fit_GD(train_data.X, train_data.y, eta=10**-5)
    cp1 = time.time()
    print("eta = 10^-5:", lin_reg.coef_)
    print("Time eta = 10^-5:", cp1 - cp0)
    print("Cost:", lin_reg.cost(train_data.X, train_data.y))
    cp0 = time.time()
    lin_reg.fit_GD(train_data.X, train_data.y, eta=10**-3)
    cp1 = time.time()
    print("eta = 10^-3:", lin_reg.coef_)
    print("Time eta = 10^-3:", cp1 - cp0)
    print("Cost:", lin_reg.cost(train_data.X, train_data.y))
    cp0 = time.time()
    lin_reg.fit_GD(train_data.X, train_data.y, eta=.05)
    cp1 = time.time()
    print("eta = .05:", lin_reg.coef_)
    print("Time eta = .05:", cp1 - cp0)
    print("Cost:", lin_reg.cost(train_data.X, train_data.y))
    cp0 = time.time()
    lin_reg.fit_GD(train_data.X, train_data.y)
    cp1 = time.time()
    print("t dependent eta:", lin_reg.coef_)
    print("Time t dependent eta:", cp1 - cp0)
    print("Cost:", lin_reg.cost(train_data.X, train_data.y))
    cp0 = time.time()
    lin_reg.fit(train_data.X, train_data.y)
    cp1 = time.time()
    print("closed form:", lin_reg.coef_)
    print("Time closed form:", cp1 - cp0)
    print("Cost:", lin_reg.cost(train_data.X, train_data.y))
    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # parts g-i: main code for polynomial regression
    print('Investigating polynomial regression...')
    train_errs = np.zeros(11)
    test_errs = np.zeros(11)
    
    for i in range(11):
        poly_reg = PolynomialRegression(m=i)
        poly_reg.fit(train_data.X, train_data.y)
        train_errs[i] = poly_reg.rms_error(train_data.X, train_data.y)
        test_errs[i] = poly_reg.rms_error(test_data.X, test_data.y)
    
    print(train_errs)
    print(test_errs)
    plot_erms(range(11), train_errs, test_errs)
        
    ### ========== TODO : END ========== ###
    
    
    print("Done!")

if __name__ == "__main__":
    main()
