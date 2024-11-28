import numpy as np
import matplotlib.pyplot as plt
import json
import GeometeryClass
import Flow
from scipy.optimize import newton
from VortexPannelMethod import VortexPannelMethod



class Main:
	"""
	A class to represent the main application logic for the airfoil project.

	Attributes
	----------
	config_file : str
		The path to the input json  file.
	radius : float
		The radius of the cylinder.
	x_low_val : float
		The lower limit for the x-axis in the plot.
	x_up_val : float
		The upper limit for the x-axis in the plot.
	geometry : GeometeryClass.Geometery
		An instance of the Geometery class.

	Methods
	-------
	load_config():
		Loads the configuration from the JSON file.
	setup_geometry():
		Initializes the Geometery object and calculates the camber, upper surface, and lower surface.
	plot():
		Plots the camber line, upper surface, and lower surface.
	surface_tangent(x):
		Calculate the surface tangent vectors at a given x-coordinate.
	surface_normal(x):
		Calculate the surface normal vectors at a given x-coordinate.
	run():
		Executes the main logic of the application.
	"""

	def __init__(self, config_file):
		"""
		Constructs all the necessary attributes for the Main object.

		Parameters
		----------
		config_file : str
			The path to the configuration file.
		"""
		self.config_file = config_file
		
		

	def load_config(self):
		"""
		Loads the configuration from the JSON file.
		"""
		with open(self.config_file, 'r') as file:
			json_vals = json.load(file)
		self.x_start = json_vals['plot_options']['x_start']
		self.x_low_val = json_vals['plot_options']['x_lower_limit']
		self.x_up_val = json_vals['plot_options']['x_upper_limit']
		self.delta_s = json_vals['plot_options']['delta_s']
		self.n_lines = json_vals['plot_options']['n_lines']
		self.delta_y = json_vals['plot_options']['delta_y']
		self.free_stream_velocity = json_vals['operating']['freestream_velocity']
		self.alpha = json_vals['operating']['alpha[deg]']
		
		airfoils = list(json_vals['airfoils'].values())
		self.airfoils = {}

		for index, element in enumerate(airfoils):
			self.airfoils[f'element{index}'] = element

		self.alpha_sweep = json_vals['alpha_sweep']
		self.run_commands = json_vals['run_commands']
		self.geometery = json_vals['geometry']

	def setup_vortex_pannel_method(self, alpha):

		self.pannelmethod = VortexPannelMethod(1.0, self.free_stream_velocity, alpha)


	def setup_Geometry(self, geometry):
		"""
		Initializes the Geometery object and calculates the camber, upper surface, and lower surface.
		"""
		self.geometry = GeometeryClass.Geometery(geometry)
		
	def load_flow_field(self, alpha):
		"""
		Loads the flow field parameters.

		Parameters:
		V_inf (float): The free stream velocity.
		alpha (float): The angle of attack.
		"""
		self.flow = Flow.Flow(self.free_stream_velocity, alpha, self.x_low_val, self.x_up_val, self.pannelmethod)

	def surface_tangent(self, x):
		"""
        Calculate the surface normal vectors at a given x-coordinate.

        Parameters:
        surface (np.ndarray): A 2xN array where the first row contains x-coordinates and the second row contains y-coordinates.
        x (float): The x-coordinate at which to calculate the surface normals.

        Returns:
        tuple: A tuple containing the unit normal vectors for the upper and lower surfaces.
        """
		delta = 1e-3  # Small perturbation for numerical differentiation

        # Interpolate to find the y-coordinate at x
		if np.abs(x-self.chord) < delta:
			y_upper_minus, y_lower_minus = self.geometry.generate_naca4_airfoil(self.NACA,x-delta)
			tangent_upper = np.array([np.abs(y_upper_minus[0]-y_upper_minus[0]), -1*(y_upper_minus[1]-y_lower_minus[1])])
			tangent_lower = np.array([np.abs(y_upper_minus[0]-y_upper_minus[0]), -1*(y_upper_minus[1]-y_lower_minus[1])])
		elif np.abs(x) < delta:
			y_upper_plus, y_lower_plus = self.geometry.generate_naca4_airfoil(self.NACA,x+delta)

			tangent_upper = np.array([-1*np.abs(y_upper_plus[0]-y_upper_plus[0]), y_upper_plus[1]-y_lower_plus[1]])
			tangent_lower = np.array([-1*np.abs(y_upper_plus[0]-y_upper_plus[0]), y_upper_plus[1]-y_lower_plus[1]])
		else:
			y_upper_plus, y_lower_plus = self.geometry.generate_naca4_airfoil(self.NACA,x+delta)

			y_upper_minus, y_lower_minus = self.geometry.generate_naca4_airfoil(self.NACA,x-delta)

			tangent_upper = np.array([np.abs(y_upper_plus[0]-y_upper_minus[0]), y_upper_plus[1] - y_upper_minus[1]])
			tangent_lower = np.array([-1 * np.abs(y_lower_plus[0]-y_lower_minus[0]), -1*(y_lower_plus[1] - y_lower_minus[1])])

			


		unit_tangent_upper = tangent_upper / np.linalg.norm(tangent_upper)	
		unit_tangent_lower = tangent_lower / np.linalg.norm(tangent_lower)	


		return unit_tangent_upper, unit_tangent_lower

	def surface_normal(self, x):
		"""""
        Calculate the surface tangent vectors at a given x-coordinate.

        Parameters:
        surface (np.ndarray): A 2xN array where the first row contains x-coordinates and the second row contains y-coordinates.
        x (float): The x-coordinate at which to calculate the surface tangents.

        Returns:
        np.ndarray: The unit tangent vector at the given x-coordinate.
        """
		unit_tangent_upper, unit_tangent_lower = self.surface_tangent(x)

		normal_upper = np.cross([unit_tangent_upper[0], unit_tangent_upper[1], 0], [0, 0, 1])
		normal_lower = np.cross([unit_tangent_lower[0], unit_tangent_lower[1], 0], [0, 0, 1])

		unit_normal_upper = normal_upper / np.linalg.norm(normal_upper)	
		unit_normal_lower = normal_lower / np.linalg.norm(normal_lower)

		return unit_normal_upper, unit_normal_lower
	
	def surface_tangential_velocity(self, x, upper=True):
		"""
		Calculate the tangential velocity at a given x-coordinate.

		Parameters:
		x (float): The x-coordinate at which to calculate the tangential velocity.

		Returns:
		tuple: A tuple containing the tangential velocity for the upper and lower surfaces.
		"""
		unit_tangent_upper, unit_tangent_lower = self.surface_tangent(x)
		unit_normal_upper, unit_normal_lower = self.surface_normal(x)

		x_geo,y_geo, _ = self.geometry.generate_naca4_airfoil(self.NACA,self.xcos)
		

		if upper:
			point, _ = self.geometry.generate_naca4_airfoil(self.NACA,x)
			point[0] += 1e-5 * unit_normal_upper[0]
			point[1] += 1e-5 * unit_normal_upper[1]
			velocity = self.flow.flow_around_an_airfoil(x_geo, y_geo, point[0], point[1], self.gamma)
			velocity_upper = np.dot(velocity, unit_tangent_upper)
			return velocity_upper
		else:
			_, point = self.geometry.generate_naca4_airfoil(self.NACA, x)
			point[0] += 1e-5 * unit_normal_lower[0]
			point[1] += 1e-5 * unit_normal_lower[1]
			velocity = self.flow.flow_around_an_airfoil(x_geo, y_geo, point[0], point[1], self.gamma)
			velocity_lower = np.dot(velocity, unit_tangent_lower)	
			return velocity_lower

	def surface_Cp(self, xcp, ycp):
		"""
		Calculate the pressure coefficient at a given x-coordinate.

		Parameters:
		x (float): The x-coordinate at which to calculate the pressure coefficient.

		Returns:
		tuple: A tuple containing the pressure coefficient for the upper and lower surfaces.
		"""
		Cp = []

		for i in range(len(xcp)):

			dx = self.x_geo[i+1] - self.x_geo[i]
			dy = self.y_geo[i+1] - self.y_geo[i]
			
			tangent = np.array([dx, dy])
			tangent = tangent / np.linalg.norm(tangent)

			normal = np.array([tangent[1], -tangent[0]])
	
	
			pointx = xcp[i] + 1e-5 * normal[0]
			pointy = ycp[i] + 1e-5 * normal[1]

			
			velocity = self.flow.flow_around_an_airfoil(self.x_geo, self.y_geo, pointx, pointy, self.gamma)
		
			velocity = np.linalg.norm(velocity)
			
			Cptemp = 1 - (velocity / self.free_stream_velocity) ** 2
			Cp.append(Cptemp)

		return Cp
	
	def stagnation_point(self):
		"""
        Calculate the stagnation point of the flow field.

        Returns:
        tuple: A tuple containing the x and y coordinates of the forward and aft stagnation points.
        """
		epsilon = 1e-5  # Small perturbation for numerical differentiation

        # Check leading edge (LE) and trailing edge (TE)
		velocity_LE = self.surface_tangential_velocity(self.LE)
		if "filename" in self.geometery:
			forward_stagnation_point = [-0.01,0]

		else:	
			if np.abs(velocity_LE) < epsilon:
				forward_stagnation_point = [self.LE, 0]
			else:
				if velocity_LE < 0:
					# Use upper surface
					upper_flag = True
					def velocity_function(x):
						return self.surface_tangential_velocity(x, upper=upper_flag)
					x_stag = self.bisection_method(velocity_function, self.LE, self.TE-0.6)
					forward_stagnation_point, _ = self.geometry.generate_naca4_airfoil(self.NACA,x_stag)
				else:
					# Use lower surface
					upper_flag = False
					def velocity_function(x):
						return self.surface_tangential_velocity(x, upper=upper_flag)
					x_stag = self.bisection_method(velocity_function, self.LE, self.TE-0.6, tol=epsilon)
					_, forward_stagnation_point = self.geometry.generate_naca4_airfoil(self.NACA,x_stag)

		aft_stagnation_point = [1, 0]

		return forward_stagnation_point, aft_stagnation_point

	def bisection_method(self,f, a, b, tol=1e-10, max_iter=100):
		"""
		Bisection method to find the root of a function f(x) = 0.
		
		Parameters:
		f: function
			The function for which the root is to be found.
		a: float
			Left endpoint of the interval.
		b: float
			Right endpoint of the interval.
		tol: float, optional
			Tolerance for the root. The method stops when |b - a| < tol.
		max_iter: int, optional
			Maximum number of iterations to perform.
		
		Returns:
		float
			The approximation for the root, or None if no root was found within the interval.
		"""
		
		# Check if a root is guaranteed in the initial interval
		if f(a) * f(b) >= 0:
			print("Bisection method fails. The function must have different signs at the endpoints.")
			return None
		
		for i in range(max_iter):
			# Calculate midpoint
			c = (a + b) / 2
			
			# Check if the function value at the midpoint is close enough to zero
			if abs(f(c)) < tol:
				print(f"Root found after {i+1} iterations: {f(c)}")
				return c
			
			# Update the interval based on the sign of f(c)
			if f(c) * f(a) < 0:
				b = c  # Root is in the left half
			else:
				a = c  # Root is in the right half
		
		print("Warning: Maximum number of iterations reached.")
		return None
	
	def plot_streamlines(self, x_start, x_lower_limit, x_upper_limit, delta_s, n_lines, delta_y):
		"""
		Plot the streamlines for the flow field.

		Parameters:
		x_start (float): The x-coordinate at which to start the streamlines.
		x_lower_limit (float): The lower limit for the x-axis.
		x_upper_limit (float): The upper limit for the x-axis.
		delta_s (float): The step size for the streamlines.
		n_lines (int): The number of streamlines to plot.
		delta_y (float): The spacing between streamlines.
		"""
		plt.figure()

		x_all = []
		y_all = []
		airfoil_geometry = {}
		
		for airfoil_key, airfoil in self.airfoils.items():
			# airfoil_geometry[airfoil_key] = {
			# 	'x': xgeo_transform,
			# 	'y': ygeo_transform,
			# 	'yc': yc,
			# 	'xcos': xcos,
			# 	'chord': airfoil['chord_length'],
			# 	'LE': airfoil['Leading_edge'],
			# 	'TE': airfoil['trailing_edge'],
			# 	'NACA': airfoil['airfoil']
			# }
			self.LE = airfoil['Leading_edge']

			# #calculate stagnation points
			#forward_stagnation_point, aft_stagnation_point = self.stagnation_point()
			# print("Forward Stagnation Point: ", forward_stagnation_point)
			# forward_normal_stag_up, forward_normal_stag_down = self.surface_normal(forward_stagnation_point[0])
			# aft_normal_stag_up, aft_normal_stag_down = self.surface_normal(aft_stagnation_point[0])
			# if forward_stagnation_point[1] < 0:
			# 	forward_normal_stag = forward_normal_stag_down
			# else:
			# 	forward_normal_stag = forward_normal_stag_up
			# if aft_stagnation_point[1] < 0:
			# 	aft_normal_stag = aft_normal_stag_down
			# else:
			# 	aft_normal_stag = aft_normal_stag_up

			# forward_stag_streamline = self.flow.streamlines(forward_stagnation_point[0]+forward_normal_stag[0] * 1e-3, forward_stagnation_point[1]+forward_normal_stag[1] * 1e-3, -1*delta_s, self.x_geo, self.y_geo, self.gamma)
			# forward_stag_streamline = np.vstack(([forward_stagnation_point[0], forward_stagnation_point[1]], forward_stag_streamline))
			# aft_stag_streamline = self.flow.streamlines(aft_stagnation_point[0]+aft_normal_stag[0] * 1e-3, aft_stagnation_point[1] + aft_normal_stag[1]*1e-3, delta_s,  self.x_geo, self.y_geo, self.gamma)
			# aft_stag_streamline = np.vstack(([aft_stagnation_point[0], aft_stagnation_point[1]], aft_stag_streamline))
			
			

		
			self.setup_Geometry(airfoil)
			
			xgeo_transform, ygeo_transform, xcos, yc = self.geometry.NACA4()

			#save the transformed coordinates to x_all and y_all as a multi-dimensional array
			x_all.append(xgeo_transform)
			y_all.append(ygeo_transform)

			
			
			plt.plot(xgeo_transform, ygeo_transform, label=f'airfoil {airfoil_key}', color='blue')

			plt.plot(xcos, yc, label=f'Camber Line {airfoil_key}', color='red')

		# Set up the plot
		
		
		plt.xlabel('X')
		plt.ylabel('Y')
		plt.title('Streamlines')
		# Calculate the streamlines
		

		for i in range(n_lines):
			print("Hold onto your seats, we're calculating those streamlines! ✈️ Predicting lift and making aerodynamic magic happen! 🚀💨")
			x = x_start
			y = -delta_y * (i+1) + 0
			streamline = self.flow.streamlines(x, y, delta_s, self.x_all_flattened, self.y_all_flattened, self.gamma, self.fake_index)
			streamline = np.vstack(([x, y], streamline))
			plt.plot(streamline[:, 0], streamline[:, 1],color='black')
			y = delta_y * (i+1) + 0
			streamline = self.flow.streamlines(x, y, delta_s, self.x_all_flattened, self.y_all_flattened, self.gamma, self.fake_index)
			streamline = np.vstack(([x, y], streamline))
			streamline = np.column_stack((streamline, np.full(streamline.shape[0], x), np.full(streamline.shape[0], y)))
			plt.plot(streamline[:, 0], streamline[:, 1],color='black')
		# plt.plot(forward_stagnation_point[0], forward_stagnation_point[1], 'ro', label='Forward Stagnation Point')
		# plt.plot(aft_stagnation_point[0], aft_stagnation_point[1], 'ro', label='Aft Stagnation Point')


		plt.xlim(min(self.x_all_flattened)-.5, max(self.x_all_flattened)+.5)
		plt.ylim(min(self.y_all_flattened)-.5, max(self.y_all_flattened)+.5)
		plt.gca().set_aspect('equal', adjustable='box')
		print("Oh, *finally*! The streamlines are done, and we've predicted the lift. I thought we’d have to wait for next year’s airshow... 🚀🙃")
		plt.show()

	def plot_cp(self, x_cp, y_cp):

		Cp = self.surface_Cp(x_cp, y_cp)
		
		plt.plot(x_cp, Cp, label='Cp Upper')
		plt.ylim(plt.ylim()[::-1])
		plt.xlim(0, 1)
		plt.xlabel('x/c')
		plt.ylabel('Pressure Coefficient')
		plt.show()

	def run(self):
		"""
		Executes the main logic of the application.
		"""
		
		self.load_config()	
		# Add the missing code to complete the run method
		airfoil_geometry = {}

		x_all = []
		y_all = []

		for airfoil_key, airfoil in self.airfoils.items():
			self.setup_Geometry(airfoil)
			xgeo_transform, ygeo_transform, xcos, yc = self.geometry.NACA4()

			#save the transformed coordinates to x_all and y_all as a multi-dimensional array
			x_all.append(xgeo_transform)
			y_all.append(ygeo_transform)

			
			airfoil_geometry[airfoil_key] = {
				'x': xgeo_transform,
				'y': ygeo_transform,
				'yc': yc,
				'xcos': xcos,
				'chord': airfoil['chord_length'],
				'LE': airfoil['Leading_edge'],
				'TE': airfoil['trailing_edge'],
				'NACA': airfoil['airfoil']
			}
			
		
		alpha = self.alpha
		# self.x_geo = airfoil_geometry['element0']['x']
		# self.y_geo = airfoil_geometry['element0']['y']
		# self.LE = airfoil_geometry['element0']['LE'][0]
		# self.chord = airfoil_geometry['element0']['chord']
		# self.NACA = airfoil_geometry['element0']['NACA']
		# self.xcos = airfoil_geometry['element0']['xcos']
		# self.yc = airfoil_geometry['element0']['yc']

		self.x_all_flattened = np.array(x_all).flatten()
		self.y_all_flattened = np.array(y_all).flatten()
		self.x_all = np.array(x_all)
		self.y_all = np.array(y_all)
		
		

		# print("x_all", x_all)
		# print("y_all", y_all)
		
		vpm = VortexPannelMethod(1.0, self.free_stream_velocity, alpha)
		self.gamma, self.fake_index = vpm.run(self.x_all, self.y_all)
		print("fake_index", self.fake_index)

		self.flow = Flow.Flow(self.free_stream_velocity, alpha, self.x_low_val, self.x_up_val, vpm)

		self.plot_streamlines(self.x_start, min(self.x_all_flattened)-.5, max(self.x_all_flattened)+.5, self.delta_s, self.n_lines, self.delta_y)
		

if __name__ == "__main__":
	main = Main('input.json')
	main.run()
	