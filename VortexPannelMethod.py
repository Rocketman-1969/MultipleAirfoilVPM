import numpy as np
class VortexPannelMethod:

    def __init__(self, chord, velocity, alpha):
        pass
        # opens json file and sets airfoil text file as an x and y array

        self.velocity = velocity
        self.alpha = alpha
        self.chord = chord
    
    
    def get_control_points(self, x, y):
        # equation (4.21)

        x_cp=(x[:-1]+x[1:])/2
        y_cp=(y[:-1]+y[1:])/2
        return x_cp, y_cp

    def get_length_of_jth_pannel(self, x, y, j):
        # sets up empty array
        #calculates length of the pannel saves as temp variable
        length_pannel=np.sqrt(np.power(x[j+1]-x[j],2)+np.power(y[j+1]-y[j],2))
        return length_pannel
    
    def get_xi_eta(self, x, y, x_cp, y_cp, l_j, j):
        xi_eta = []

        matrix1 = np.array([[x[j+1]-x[j], y[j+1]-y[j]], [-(y[j+1]-y[j]), x[j+1]-x[j]]])
        matrix2 = np.array([x_cp-x[j], y_cp-y[j]])

        xi_eta =(1/l_j) * np.matmul(matrix1, matrix2)
        xi = xi_eta[0]
        eta = xi_eta[1]

        return xi, eta
    
    def get_phi(self, eta, xi,l_j):
        phi=np.arctan2(eta * l_j, eta**2 + xi**2 - xi * l_j)
        return phi
    
    def get_psi(self,eta,xi,l_j):
        psi=(1/2)*np.log((xi**2 + eta**2)/(((xi-l_j)**2)+eta**2))
        return psi
    
    def get_P_matrix(self, x, y, x_cp, y_cp, i, j):
        l_j = self.get_length_of_jth_pannel(x, y, j)
        
        xi, eta = self.get_xi_eta(x, y, x_cp, y_cp, l_j, j)
        
        phi = self.get_phi(eta, xi, l_j)

        psi = self.get_psi(eta, xi, l_j)

        matrix1 = np.array([[(x[j+1]-x[j]), -1*(y[j+1]-y[j])],
        [(y[j+1]-y[j]), (x[j+1]-x[j])]])
        matrix2 = np.array([[((l_j-xi)*phi+eta*psi), (xi*phi-eta*psi)],[(eta*phi-(l_j-xi)*psi-l_j), (-eta*phi-xi*psi+l_j)]])
        P = (1/(2*np.pi*l_j**2))*np.matmul(matrix1, matrix2)    
        return P
    
    def get_A_matrix(self, x, y, x_cp, y_cp):
        x = x.flatten()
        y = y.flatten()

        kutta_start_index = 0

        A = np.zeros((len(x), len(x)))
        #ith control point
        for i in range(len(x)-1):
            # skip fake indices
            
            if i in self.fake_indices:
                A[i,kutta_start_index] = 1.0
                A[i,i] = 1.0
                kutta_start_index = i+1
                continue
            #jth pannel
            for j in range(len(x)-1):
                P = self.get_P_matrix(x, y, x_cp[i], y_cp[i], i, j)

                l_i = self.get_length_of_jth_pannel(x, y, i)

                A[i,j]=A[i,j]+((x[i+1]-x[i])/l_i)*P[1,0]-((y[i+1]-y[i])/l_i)*P[0,0]
                A[i,j+1]=A[i,j+1]+((x[i+1]-x[i])/l_i)*P[1,1]-((y[i+1]-y[i])/l_i)*P[0,1]
        # apply the kutta condition when fake indices are reached
        A[-1,kutta_start_index] = 1.0
        A[-1,-1] = 1.0
        
        return A
    
    def get_B_matrix(self, x, y):
        B = np.zeros((len(x),1))
        alpha = np.deg2rad(self.alpha)
        for i in range(len(x)-1):
            if i in self.fake_indices:
                B[i,0] = 0.0
                continue
            l_j=self.get_length_of_jth_pannel(x,y,i)
            B[i,0]=self.velocity *(((y[i+1]-y[i])*np.cos(alpha)-(x[i+1]-x[i])*np.sin(alpha))/l_j)
        B[-1,0]=0.0

        return B
    
    def get_gamma(self, A, B):
        gamma = np.linalg.solve(A,B)
        return gamma
    
    def get_CL(self, gamma, x, y):
        CL = 0
        for i in range(len(x)-1):
            l_i = self.get_length_of_jth_pannel(x, y, i)
            CL += (l_i/self.chord)*((gamma[i] + gamma[i+1])/self.velocity)
        return CL
    
    def get_cmle(self, gamma, x, y):
        Cmle_temp = 0
        for i in range(len(x)-1):
            l_i = self.get_length_of_jth_pannel(x, y, i)
            Cmle_temp+=(l_i/self.chord)*(((2*x[i]*gamma[i]+x[i]*gamma[i+1]+x[i+1]*gamma[i]+2*x[i+1]*gamma[i+1])/(self.chord*self.velocity))*np.cos(np.deg2rad(self.alpha))+((2*y[i]*gamma[i]+y[i]*gamma[i+1]+y[i+1]*gamma[i]+2*y[i+1]*gamma[i+1])/(self.chord*self.velocity))*np.sin(np.deg2rad(self.alpha)))
        Cmle=-(1/3)*Cmle_temp
        return Cmle
    
    def get_Cmc4(self, CL, Cmle):

        Cmc4 = Cmle + CL/4 * np.cos(np.deg2rad(self.alpha))

        return Cmc4
    

    def run(self, x_all, y_all):
        print("running VPM")
        num_airfoils = x_all.shape[0]
        airfoil_lengths = []
        x_cp = []
        y_cp = []
        print("num_airfoils: ", num_airfoils)
        for i in range(num_airfoils):
            x = x_all[i]
            y = y_all[i]
            airfoil_length = len(x)
            airfoil_lengths.append(airfoil_length)
            
        x_all = x_all.flatten()
        y_all = y_all.flatten()

        x_cp,y_cp = self.get_control_points(x_all, y_all)

        self.fake_indices = []
        offset = 0
        for length in airfoil_lengths:
            self.fake_indices.append(offset + length-1)
            offset += length
        
    
        A = self.get_A_matrix(x_all, y_all, x_cp, y_cp)
        print("A: ", A)
        B = self.get_B_matrix(x_all, y_all)
        print("B: ", B)
        gamma = self.get_gamma(A, B)
        print("gamma: ", gamma)

        return gamma, self.fake_indices

    
    # def run(self, x, y):

    #     x_cp, y_cp = self.get_control_points(x, y)
        
    #     A = self.get_A_matrix(x, y, x_cp, y_cp)

    #     B = self.get_B_matrix(x, y)
       
    #     gamma = self.get_gamma(A, B)

    #     C_L = self.get_CL(gamma, x, y)

    #     Cmle = self.get_cmle(gamma, x, y)

    #     Cmc4 = self.get_Cmc4(C_L, Cmle)

    #     return C_L, Cmle, Cmc4, x_cp, y_cp, gamma