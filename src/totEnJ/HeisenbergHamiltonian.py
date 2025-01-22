import numpy as np

class HeisenbergHamiltonian:
    """The Heisenberg magnetic interactions will be fully defined by this model.
    The object will hold a symbolic vector of all the free parameters (ground-state energy E0, (anisotropic) exchange interactions, single-site anisotropy, DMI constants)
    Given two spins, the model will return a vector of prefactors for all these parameters.  
    """
    def __init__(self, type='scalar'):
        """Define what the 'energy' looks like.

        Args:
            type (str, optional): Type of the Hamiltonian. Defaults to 'scalar'.

        Raises:
            ValueError: if the type is not one of the available types.

        Returns:
            list of floats: a vector of prefactors of the Hamiltonian parameters between two interacting spins.
        """
        self.type = type

        available_types = ['scalar', 'XXZ', 'tensorial']

        if type == 'scalar':
            self.two_site_parameters = ['J', 'D']
            def two_site_parameter_prefactors(spin1=(1,0,0), spin2=(1,0,0)):
                spin1 = np.array(spin1)
                spin2 = np.array(spin2)
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! check expression for DMI 
                return [np.dot(spin1, spin2), np.sum(np.cross(spin1, spin2))]
            self.two_site_energy = two_site_parameter_prefactors

        elif type == 'XXZ':
            self.two_site_parameters = ['J_xx/yy', 'J_zz', 'D_x/y', 'D_z']
            def two_site_parameter_prefactors(spin1=(1,0,0), spin2=(1,0,0)):
                spin1 = np.array(spin1)
                spin2 = np.array(spin2)
                D_xyz = np.cross(spin1, spin2)
                return [(spin1[0]*spin2[0]+spin1[1]*spin2[1])/2, spin1[2]*spin2[2], (D_xyz[0]+D_xyz[1])/2, D_xyz[2]]
            self.two_site_energy = two_site_parameter_prefactors

        elif type == 'tensorial':
            self.two_site_parameters = ['J_xx', 'J_xy', 'J_xz', 'J_yx', 'J_yy', 'J_yz', 'J_zx', 'J_zy', 'J_zz', 'D_x', 'D_y', 'D_z']
            def two_site_parameter_prefactors(spin1=(1,0,0), spin2=(1,0,0)):
                spin1 = np.array(spin1)
                spin2 = np.array(spin2)
                return [spin1[0]*spin2[0], spin1[0]*spin2[1], spin1[0]*spin2[2], spin1[1]*spin2[0], spin1[1]*spin2[1], spin1[1]*spin2[2], spin1[2]*spin2[0], spin1[2]*spin2[1], spin1[2]*spin2[2], *np.cross(spin1, spin2)]
            self.two_site_energy = two_site_parameter_prefactors

        else:
            raise ValueError(f'Invalid type of Hamiltonian. Please choose from {', '.join(available_types)}.')
        
        self.single_site_parameters = ['K']
        def single_site_parameter_prefactors(spin=(1,0,0)):
            spin = np.array(spin)
            return [spin[2]**2-spin[0]**2-spin[1]**2]
        self.single_site_energy = single_site_parameter_prefactors

    def __str__(self):
        return f'Heisenberg Hamiltonian with {self.type} interactions: the two-site parameters are {self.two_site_parameters.join(', ')}.'

    def print_two_site_parameters(self):
        print(f'The two-site parameters are {self.two_site_parameters.join(', ')}.')

    def print_two_site_energy(self, spin1=(1,0,0), spin2=(1,0,0)):
        print(f'The two-site energy prefactors are {self.energy(spin1, spin2).join(', ')}.')

    def get_total_energy(self, magnetic_moments, site_spins, site_labels, site_multiplicity):
        """
        Return an array structured as [site_i, its_neighbor_j, interaction_prefactors], 
            one for two-site interactions and one for single-site interactions.
            (For single-site interactions, the spin of site_i does not matter, only the spin of its neighbor.)

        All the below arrays have the same dimensions (N_atoms_magnetic_unit_cell, N_neighbors_up_to_cutoff) 
        as all_neighbors_for_all_sites = get_all_neighbors(neighbor_cutoff).

        Args:
            magnetic_moments (1D-array of spins of atoms in the magnetic unit cell):
            site_spins (2D-array-like, see above for dimensions): 
            site_labels (2D-array-like, see above for dimensions): 
            site_multiplicity (2D-array-like, see above for dimensions): all neighbors should be considered equal if Hamiltonian is isotropic
        """
        print('type(site_spins):', type(site_spins))
        assert site_spins.shape[:2] == site_labels.shape == site_multiplicity.shape, f'The number of spins, labels and multiplicities must be the same, but they are {site_spins.shape[:2]}, {site_labels.shape}, and {site_multiplicity.shape}.'

        N_atoms_in_magnetic_unit_cell, N_neighbors = site_labels.shape
        N_two_site_parameters = len(self.two_site_parameters)
        N_single_site_parameters = len(self.single_site_parameters)

        # ---- TWO-SITE INTERACTIONS ----
        two_site_prefactors = np.zeros((N_atoms_in_magnetic_unit_cell, N_neighbors, N_two_site_parameters))
        for i in range(N_atoms_in_magnetic_unit_cell):
            for j in range(N_neighbors):
                # the i-th atom (with spin = magnetic_moments[i]) interacts with its j-th neighbor (with spin = site_spins[i,j])
                two_site_prefactors[i,j,:] = site_multiplicity[i,j] * np.array(self.two_site_energy(magnetic_moments[i], site_spins[i,j]))

        # ---- SINGLE-SITE INTERACTIONS ----
        single_site_prefactors = np.zeros((N_atoms_in_magnetic_unit_cell, N_neighbors, N_single_site_parameters))
        for i in range(N_atoms_in_magnetic_unit_cell):
            for j in range(N_neighbors):
                single_site_prefactors[i,j,:] = site_multiplicity[i,j] * np.array(self.single_site_energy(site_spins[i,j]))

        self.two_site_prefactors = two_site_prefactors
        self.single_site_prefactors = single_site_prefactors

        print('single_site_prefactors.shape:', single_site_prefactors.shape)
        print('two_site_prefactors:', two_site_prefactors.shape)

        # DECIDE if the neighbor is in the magnetic unit cell or not
        # if in the supercell, the interaction would be calculated twice!!!
        # --> calculate only if the first site is smaller than the second:
        # if all_neighbors[i][j].image == np.array([0,0,0]) and i < j:
            # consider for the total energy calculation

        # my neighbor = all_neighbors_for_all_sites[i][j] is the neighbor j of an atom i
        # there are as many i as there are atoms in the magnetic unit cell
        # there are as many j as there are neighbors of atom i, given by 'neighbor_cutoff'
        # my_neighbor = all_neighbors_for_all_sites[0][17]
        # # the atom label:
        # print('PeriodicNeighbor.species:', my_neighbor.species)
        # # the unit cell coordinates:
        # print('PeriodicNeighbor.image:', my_neighbor.image)
        # # the index of the atom in the magnetic unit cell:
        # print('PeriodicNeighbor.index:', my_neighbor.index)
        # # the (x,y,z) coordinates in space (in Angstroms):
        # print('PeriodicNeighbor.coords:', my_neighbor.coords)
        
        


        
        