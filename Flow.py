import numpy as np
from VortexPannelMethod import VortexPannelMethod as vpm

class Flow:

    def __init__(self, V_inf, alpha, x_low_val, x_up_val, vortex_pannel_method):
        self.V_inf = V_inf
        self.alpha = alpha
        self.x_low_val = x_low_val
        self.x_up_val = x_up_val
        self.vpm = vortex_pannel_method

    def flow_around_an_airfoil(self, x, y, x_arb, y_arb, gamma, fake_index):
        alpha = np.deg2rad(0)
        P=[]
        Vx = self.V_inf*np.cos(alpha)
        Vy = self.V_inf*np.sin(alpha)
        
        for i in range(len(x)-1):

            if i in fake_index:
                
                continue
    
            P= self.vpm.get_P_matrix(x, y, x_arb, y_arb, i, i)

            Vx += gamma[i]*P[0,0]+gamma[i+1]*P[0,1]
            Vy += gamma[i]*P[1,0]+gamma[i+1]*P[1,1]
        

        velocity = np.array([Vx[0], Vy[0]])

        return velocity
    
    def unit_velocity(self, x_arb, y_arb, x_geo, y_geo, gamma, fake_index):
        velocity = self.flow_around_an_airfoil(x_geo, y_geo, x_arb, y_arb, gamma, fake_index)
        velocity = velocity/np.linalg.norm(velocity)
        
        return velocity
    
    def streamlines(self, x, y, delta_s, x_geo, y_geo, gamma, fake_index, tol=1e-11):
        """
        Calculate the streamlines at a given x-coordinate using RK45 with adaptive step sizing.
        
        Parameters:
        x (float): The initial x-coordinate.
        y (float): The initial y-coordinate.
        delta_s (float): Initial step size for the streamlines.
        x_geo, y_geo: Geometry arrays for influence calculations.
        gamma: Circulation strength array.
        fake_index: Index for excluding self-influence.
        tol (float): Tolerance for adaptive step sizing.
        
        Returns:
        np.array: Streamlines as a 2D array with coordinates [[x1, y1], [x2, y2], ...].
        """
        streamline = []
        iter = 0
        h = delta_s

        while True:
            # RK45 coefficients
            c = [0, 1/4, 3/8, 12/13, 1, 1/2]
            a = [
                [],
                [1/4],
                [3/32, 9/32],
                [1932/2197, -7200/2197, 7296/2197],
                [439/216, -8, 3680/513, -845/4104],
                [-8/27, 2, -3544/2565, 1859/4104, -11/40]
            ]
            b4 = [25/216, 0, 1408/2565, 2197/4104, -1/5, 0]  # Fourth-order solution
            b5 = [16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55]  # Fifth-order solution

            # Compute k-values
            k = []
            for i in range(len(c)):
                x_temp = x + h * sum(a[i][j] * k[j][0] for j in range(len(k))) if i > 0 else x
                y_temp = y + h * sum(a[i][j] * k[j][1] for j in range(len(k))) if i > 0 else y
                k.append(self.unit_velocity(x_temp, y_temp, x_geo, y_geo, gamma, fake_index))

            # Fourth and fifth-order solutions
            x4 = x + h * sum(b4[i] * k[i][0] for i in range(len(k)))
            y4 = y + h * sum(b4[i] * k[i][1] for i in range(len(k)))
            x5 = x + h * sum(b5[i] * k[i][0] for i in range(len(k)))
            y5 = y + h * sum(b5[i] * k[i][1] for i in range(len(k)))

            # Error estimation
            error = max(abs(x5 - x4), abs(y5 - y4))

            # Check if error is acceptable
            if error <= tol:
                # Accept the step
                x, y = x5, y5
                streamline.append([x, y])

            # Adjust step size
            if error == 0:
                h *= 2  # Avoid division by zero
            else:
                h *= 0.9 * (tol / error) ** 0.2  # Update step size with safety factor

            # Break conditions
            if x < self.x_low_val or x > self.x_up_val:
                break

            iter += 1
            if iter > 200:  # Optional: safeguard against infinite loops
                print("Streamline iteration limit reached.")
                break

        return np.array(streamline)
