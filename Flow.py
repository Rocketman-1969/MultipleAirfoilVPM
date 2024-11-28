import numpy as np
from VortexPannelMethod import VortexPannelMethod as vpm

class Flow:

    def __init__(self, V_inf, alpha, x_low_val, x_up_val, vortex_pannel_method=None):
        self.V_inf = V_inf
        self.alpha = alpha
        self.x_low_val = x_low_val
        self.x_up_val = x_up_val
        self.vpm = vortex_pannel_method

    def flow_around_an_airfoil(self, x, y, x_arb, y_arb, gamma, fake_index):
        alpha = np.deg2rad(self.alpha)
        P=[]
        Vx = self.V_inf*np.cos(alpha)
        Vy = self.V_inf*np.sin(alpha)
        for i in range(len(x)-1):

            if i in fake_index:
                continue
    
            P= self.vpm.get_P_matrix(x, y, x_arb, y_arb, i, j=i)

            Vx += gamma[i]*P[0,0]+gamma[i+1]*P[0,1]
            Vy += gamma[i]*P[1,0]+gamma[i+1]*P[1,1]

        velocity = np.array([Vx[0], Vy[0]])

        return velocity
    
    def unit_velocity(self, x_arb, y_arb, x_geo, y_geo, gamma, fake_index):
        velocity = self.flow_around_an_airfoil(x_geo, y_geo, x_arb, y_arb, gamma, fake_index)
        velocity = velocity/np.linalg.norm(velocity)
        
        return velocity
    
    def streamlines(self, x, y, delta_s, x_geo, y_geo, gamma, fake_index):
        """
        Calculate the streamlines at a given x-coordinate.
    
        Parameters:
        x (float): The x-coordinate at which to calculate the streamlines.
        delta_s (float): The step size for the streamlines.
    
        Returns:
        tuple: A tuple containing the streamlines for the upper and lower surfaces.
        """
        streamline = []
        iter = 0
        
        while True:
            k1 = delta_s * self.unit_velocity(x, y, x_geo, y_geo, gamma, fake_index)
            k2 = delta_s * self.unit_velocity(x + 0.5*k1[0], y + 0.5*k1[1], x_geo, y_geo, gamma, fake_index)
            k3 = delta_s * self.unit_velocity(x + 0.5*k2[0], y + 0.5*k2[1], x_geo, y_geo, gamma, fake_index)
            k4 = delta_s * self.unit_velocity(x + k3[0], y + k3[1], x_geo, y_geo, gamma, fake_index)

            x_new = x + (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6
            y_new = y + (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6
            
            streamline.append([x_new, y_new])

            x = x_new
            y = y_new

            if x_new < self.x_low_val or x_new > self.x_up_val:
                
                break
            if iter > 150:
                print("Streamline iteration limit reached.")
                break
            iter += 1

            
        return np.array(streamline)