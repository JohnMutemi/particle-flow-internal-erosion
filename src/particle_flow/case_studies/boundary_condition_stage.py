"""
Boundary condition application stage for tunnel water inrush simulation.
"""


class BoundaryConditionStage:
    def __init__(self, config, coupling, water_pressure, tunnel_length, tunnel_diameter):
        self.config = config
        self.coupling = coupling
        self.water_pressure = water_pressure
        self.tunnel_length = tunnel_length
        self.tunnel_diameter = tunnel_diameter

    def initialize_tunnel_geometry(self):
        length_scale = self.config['simulation']['domain_size'][0] / \
            self.tunnel_length
        diameter_scale = self.config['simulation']['domain_size'][1] / \
            self.tunnel_diameter
        tunnel_params = {
            'length': min(self.tunnel_length * length_scale, self.config['simulation']['domain_size'][0]),
            'diameter': min(self.tunnel_diameter * diameter_scale, min(self.config['simulation']['domain_size'][1:])),
            'center': [min(self.tunnel_length * length_scale / 2, self.config['simulation']['domain_size'][0] / 2), 0.0, 0.0],
            'roughness': 0.001
        }
        boundary_conditions = {
            'inlet': {
                'type': 'pressure',
                'value': self.water_pressure,
                'position': 0.0
            },
            'outlet': {
                'type': 'pressure',
                'value': 0.0,
                'position': min(self.tunnel_length * length_scale, self.config['simulation']['domain_size'][0])
            },
            'walls': {
                'type': 'no-slip',
                'roughness': 0.001
            }
        }
        self.coupling.cfd_solver.set_tunnel_geometry(tunnel_params)
        self.coupling.cfd_solver.set_boundary_conditions(boundary_conditions)
