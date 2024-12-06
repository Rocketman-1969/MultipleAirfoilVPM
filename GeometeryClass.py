import numpy as np


class Geometery:
   

    def __init__(self, airfoil):
        
        self.NACA = airfoil['airfoil']
        self.n_points = airfoil['n_points']
        self.chord = airfoil['chord_length']
        self.LE = airfoil['Leading_edge']
        mounting_angle = airfoil['mounting_angle[deg]']
        self.mounting_angle = np.deg2rad(mounting_angle)
    
    
    def Cose_cluster(self):
        n_points = self.n_points
            # Define step size for odd or even number of points
        if n_points % 2 == 1:  # Odd number of points
            # Equation (4.2.16): Calculate delta_theta
            delta_theta = np.pi / (n_points // 2)
            indices = np.arange(1, n_points // 2 + 1)
            x_cos = 0.5 * (1 - np.cos(indices * delta_theta))
            x_cos = np.insert(x_cos, 0, 0.0)
        else:  # Even case
             # Equation (4.2.19): Calculate delta_theta
            delta_theta = np.pi / (n_points / 2 - 0.5)
            indices = np.arange(1, n_points // 2 + 1)
            x_cos = 0.5 * (1 - np.cos(indices * delta_theta - 0.5 * delta_theta))
            
        return x_cos


    def generate_naca4_airfoil(self):
        # Convert naca to string if it's an integer
        naca = str(self.NACA)
        x = self.x_cos
        
        # Extract NACA parameters
        
        m = int(naca[0]) / 100.0  # Maximum camber
        p = int(naca[1]) / 10.0   # Position of maximum camber

        # Camber line
        if m == 0:
            yc = np.zeros_like(x)
            dyc_dx = np.zeros_like(x)
        else:
            yc = np.where(x < p, m / (p**2) * (2 * p * x - x**2), m / ((1 - p)**2) * ((1 - 2 * p) + 2 * p * x - x**2))
            dyc_dx = np.where(x < p, 2 * m / (p**2) * (p - x), 2 * m / ((1 - p)**2) * (p - x))


        t = int(naca[2:]) / 100.0 # Thickness
        
        # Thickness distribution
        yt = t/2 * (2.980 * np.sqrt(x) - 1.320 * x - 3.286 * x**2 + 2.441 * x**3 - 0.815 * x**4)
        
        # Angle of the camber line
        theta = np.arctan(dyc_dx)

        # Upper and lower surface coordinates for each x value
        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)

        # If a single x value was provided, return just the corresponding upper and lower surface points
        if x.size == 1:
            return np.array([xu[0], yu[0]]), np.array([xl[0], yl[0]])

        # Combine coordinates for the full airfoil, including both leading and trailing edges
        elif self.n_points % 2 == 1:
            x_coords = np.concatenate([xl[::-1], xu[1:]])
            y_coords = np.concatenate([yl[::-1], yu[1:]])
            
        else:
            x_coords = np.concatenate([xl[::-1], xu])
            y_coords = np.concatenate([yl[::-1], yu])

        return x_coords, y_coords, yc
    
    def NACA4(self):
        
        self.x_cos = self.Cose_cluster()
        x_geo, y_geo, yc = self.generate_naca4_airfoil()
        x_geo_transform = x_geo * self.chord
        y_geo_transform = y_geo * self.chord
        self.x_cos = self.x_cos * self.chord
        yc = yc * self.chord

        R = np.array([[np.cos(self.mounting_angle), np.sin(self.mounting_angle)], [-np.sin(self.mounting_angle), np.cos(self.mounting_angle)]])
        coords = np.vstack([x_geo_transform, y_geo_transform])
        camber = np.vstack([self.x_cos, yc])
        transformed_coords = R @ coords
        transformed_camber = R @ camber
        x_geo_transformed = transformed_coords[0, :] + self.LE[0]
        y_geo_transformed = transformed_coords[1, :] + self.LE[1]
        self.x_cos = transformed_camber[0, :] + self.LE[0]
        yc = transformed_camber[1, :] + self.LE[1]

        return x_geo_transformed, y_geo_transformed, self.x_cos, yc