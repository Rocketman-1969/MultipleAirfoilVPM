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

		self.delta_s = json_vals['plot_options']['delta_s']
		self.n_lines = json_vals['plot_options']['n_lines']
		self.delta_y = json_vals['plot_options']['delta_y']

		self.free_stream_velocity = json_vals['operating']['freestream_velocity']
		self.alpha = json_vals['operating']['alpha[deg]']
		
		airfoils = list(json_vals['airfoils'].values())
		self.airfoils = {}
		for index, element in enumerate(airfoils):
			self.airfoils[f'element{index}'] = element
			
		

		self.geometery = json_vals['geometry']

	def setup_vortex_pannel_method(self):

		self.vpm = VortexPannelMethod(1.0, self.free_stream_velocity, self.alpha)

	def setup_Geometry(self, geometry):
		"""
		Initializes the Geometery object and calculates the camber, upper surface, and lower surface.
		"""
		self.geometry = GeometeryClass.Geometery(geometry)
		
	def load_flow_field(self, x_low_val, x_up_val):
		"""
		Loads the flow field parameters.

		Parameters:
		V_inf (float): The free stream velocity.
		alpha (float): The angle of attack.
		"""
		self.flow = Flow.Flow(self.free_stream_velocity, self.alpha, x_low_val, x_up_val, self.vpm)

	def plot_streamlines(self, x_low_val, x_up_val, y_avg):
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

		n_lines = self.n_lines
		delta_s = self.delta_s
		delta_y = self.delta_y
		

		x_all = []
		y_all = []
				
		for airfoil_key, airfoil in self.airfoil_geometry.items():

			#save the transformed coordinates to x_all and y_all as a multi-dimensional array
			x_all.append(airfoil['x'])
			y_all.append(airfoil['y'])

			plt.plot(airfoil['x'], airfoil['y'], label=f'airfoil {airfoil_key}', color='blue')

			plt.plot(airfoil['xcos'], airfoil['yc'], label=f'Camber Line {airfoil_key}', color='red')

		# Set up the plot
		plt.xlabel('X')
		plt.ylabel('Y')
		plt.title('Streamlines')
		# Calculate the streamlines
		print("Hold onto your seats, we're calculating those streamlines! ✈️ Predicting lift and making aerodynamic magic happen! 🚀💨")
		x = x_low_val
		y = y_avg
		streamline = self.flow.streamlines(x, y, delta_s, self.x_all_flattened, self.y_all_flattened, self.gamma, self.fake_index)
		streamline = np.vstack(([x, y], streamline))
		plt.plot(streamline[:, 0], streamline[:, 1],color='black')
		for i in range(n_lines):
			print("Hold onto your seats, we're calculating those streamlines! ✈️ Predicting lift and making aerodynamic magic happen! 🚀💨")
			x = x_low_val
			y = -delta_y * (i+1) + y_avg
			streamline = self.flow.streamlines(x, y, delta_s, self.x_all_flattened, self.y_all_flattened, self.gamma, self.fake_index)
			streamline = np.vstack(([x, y], streamline))
			plt.plot(streamline[:, 0], streamline[:, 1],color='black', linewidth=0.5)
			print("Hold onto your seats, we're calculating those streamlines! ✈️ Predicting lift and making aerodynamic magic happen! 🚀💨")
			y = delta_y * (i+1) + y_avg
			streamline = self.flow.streamlines(x, y, delta_s, self.x_all_flattened, self.y_all_flattened, self.gamma, self.fake_index)
			streamline = np.vstack(([x, y], streamline))
			streamline = np.column_stack((streamline, np.full(streamline.shape[0], x), np.full(streamline.shape[0], y)))
			plt.plot(streamline[:, 0], streamline[:, 1],color='black', linewidth=0.5)

		plt.xlim(x_low_val, x_up_val)
		plt.ylim(min(self.y_all_flattened)-.5, max(self.y_all_flattened)+.5)
		plt.gca().set_aspect('equal', adjustable='box')
		print("Oh, *finally*! The streamlines are done, and we've predicted the lift. I thought we’d have to wait for next year’s airshow... 🚀🙃")
		plt.show()

	def run(self):
		"""
		Executes the main logic of the application.
		"""
		# Load the configuration
		self.load_config()	
		
		# Initialize the Geometery object and calculate the camber, upper surface, and lower surface
		self.airfoil_geometry = {}
		x_all = []
		y_all = []

		for airfoil_key, airfoil in self.airfoils.items():
			self.setup_Geometry(airfoil)
			xgeo_transform, ygeo_transform, xcos, yc = self.geometry.NACA4()

			#save the transformed coordinates to x_all and y_all as a multi-dimensional array
			x_all.append(xgeo_transform)
			y_all.append(ygeo_transform)

			self.airfoil_geometry[airfoil_key] = {
				'x': xgeo_transform,
				'y': ygeo_transform,
				'yc': yc,
				'xcos': xcos,
				'chord': airfoil['chord_length'],
				'LE': airfoil['Leading_edge'],
				'NACA': airfoil['airfoil']
			}
			
		# Define global airfoil geometery
		self.x_all_flattened = np.array(x_all, dtype=object)
		self.y_all_flattened = np.array(y_all, dtype=object)

		self.x_all_flattened = np.concatenate(self.x_all_flattened)
		self.y_all_flattened = np.concatenate(self.y_all_flattened)

		self.x_all = np.array(x_all)
		self.y_all = np.array(y_all)

		# Calculate the lower and upper limits for the x-axis
		x_low_val = min(self.x_all_flattened)-1.0
		x_up_val = max(self.x_all_flattened)+1.0
		y_avg = np.mean(self.y_all_flattened)
		
		# Set up and Run the Vortex Pannel Method
		self.setup_vortex_pannel_method()
		self.gamma, self.fake_index = self.vpm.run(self.x_all, self.y_all)
		
		# Set up the Flow Field

		self.load_flow_field(x_low_val, x_up_val)

		# Plot the Streamlines
		self.plot_streamlines(x_low_val, x_up_val, y_avg)
		
if __name__ == "__main__":
	main = Main('input.json')
	main.run()
	