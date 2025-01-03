import numpy as np
class VortexPannelMethod:

    def __init__(self, velocity, alpha):
        
        self.velocity = velocity
        self.alpha = alpha
    
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
        l_j = self.get_length_of_jth_pannel(x, y, j) 
        matrix1 = np.array([[x[j+1]-x[j], y[j+1]-y[j]], [-1*(y[j+1]-y[j]), x[j+1]-x[j]]])
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
    
    def get_P_matrix(self, x, y, x_cp, y_cp, j):
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
        # x = x.flatten()
        # y = y.flatten()

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
                if j in self.fake_indices:
                    continue
                P = self.get_P_matrix(x, y, x_cp[i], y_cp[i], j)

                l_i = self.get_length_of_jth_pannel(x, y, i)

                A[i,j]=A[i,j]+((x[i+1]-x[i])/l_i)*P[1,0]-((y[i+1]-y[i])/l_i)*P[0,0]
                A[i,j+1]=A[i,j+1]+((x[i+1]-x[i])/l_i)*P[1,1]-((y[i+1]-y[i])/l_i)*P[0,1]

        # apply the kutta condition when fake indices are reached
        A[-1,kutta_start_index] = 1.0
        A[-1,-1] = 1.0
        
        return A
    
    def get_B_matrix(self, x, y):
        B = np.zeros((len(x)))
        alpha = np.deg2rad(self.alpha)
        
        for i in range(len(x)-1):
            if i in self.fake_indices:
                B[i] = 0.0
                continue
            l_j=self.get_length_of_jth_pannel(x,y,i)
            B[i]=self.velocity *(((y[i+1]-y[i])*np.cos(alpha)-(x[i+1]-x[i])*np.sin(alpha))/l_j)
        B[-1]=0.0

        return B
    
    def get_gamma(self, A, B):
        gamma = np.linalg.solve(A,B)
        
        return gamma
    
    def find_fake_indices(self, airfoil_lengths):
        self.fake_indices = []
        offset = 0
        for length in airfoil_lengths:
            self.fake_indices.append(offset + length-1)
            offset += length
    
    def get_CL(self, gamma, x, y, chord):
        CL = 0
        for i in range(len(x)-1):
            l_i = self.get_length_of_jth_pannel(x, y, i)
            CL += (l_i/chord)*((gamma[i] + gamma[i+1])/self.velocity)
        return CL
        
    def run(self, x_all, y_all, chord):
        print("running VPM")
        num_airfoils = x_all.shape[0]
        airfoil_lengths = []
        x_cp = []
        y_cp = []

        for i in range(num_airfoils):
            x = x_all[i]
            y = y_all[i]
            airfoil_length = len(x)
            airfoil_lengths.append(airfoil_length)
            
        x_all = np.concatenate(x_all)
        y_all = np.concatenate(y_all)

        x_cp,y_cp = self.get_control_points(x_all, y_all)

        self.find_fake_indices(airfoil_lengths)
        
        A = self.get_A_matrix(x_all, y_all, x_cp, y_cp)
        print("A matrix calculated", A)

        B = self.get_B_matrix(x_all, y_all)

        gamma = self.get_gamma(A, B)

        index = 0
        CL_total = 0
        CL=np.array([])
        for i in range(num_airfoils):
            CL_airfoil_x = x_all[index:self.fake_indices[i]]
            CL_airfoil_y = y_all[index:self.fake_indices[i]]
            CL_gamma = gamma[index:self.fake_indices[i]]
            CL_chord = chord[i]

            CL_temp = self.get_CL(CL_gamma, CL_airfoil_x, CL_airfoil_y, CL_chord)
            CL_total += CL_temp
            CL = np.append(CL, CL_temp)

            index = self.fake_indices[i] + 1
        
        CL = np.append(CL, CL_total)

        return gamma, self.fake_indices, CL
