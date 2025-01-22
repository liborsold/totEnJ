from pymatgen.core import Lattice, Structure, Molecule, PeriodicNeighbor
from pymatgen.vis.structure_vtk import StructureVis
from os.path import exists
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from matplotlib.patches import Circle, Polygon
from totEnJ.utils import nn_order_from_distances, count_nn_order_neighbors
from totEnJ.HeisenbergHamiltonian import HeisenbergHamiltonian

class StructureJ(Structure):
    """Class for the magnetic structure analysis and calculation of the total energy using the Heisenberg Hamiltonian, derived from pymatgen's Structure.
    """
    
    def initialize(self, magnetic_atoms=None, discard_nonmagnetic_atoms=None, magnetic_supercell=None, 
                   show_supercell=None, supercell_out_name=None, Heisenberg_Hamiltonian_type=None, 
                   magnetic_moments=None, neighbor_cutoff=None, round_decimals=None, 
                   keep_only_first_unique_NN_label_combo=True):
        """Ideally this would be part of __init__ but there seems to be problem with overriding the 
            pymatgen's .from_file() constructor. So, this is a workaround.

        Args:
            magnetic_atoms (list of integers, optional): Indeces of the magnetic atoms taking part in the Heisenberg exchange. Defaults to None.
            discard_nonmagnetic_atoms (bool, optional): Discard nonmagnetic atoms from the structure right at the beginning. Defaults to None.
            magnetic_supercell (tuple of three integers, optional): Dimensions of the supercell to be created. Defaults to None.
            supercell_out_name (string, optional): path + file name of the supercell file to be saved. Defaults to None.
            magnetic_moments (2D numpy array, optional): (N_atoms_in_magnetic_unit_cell, 3) dimensional array of floats indicating the magnetic moment (spin direction) for each atom in the magnetic unit cell. Defaults to None.
            neighbor_cutoff (float, optional): Radius in Angstrom around each atom within which the neighbors will be considered. Defaults to None.
            round_decimals (integer, optional): to what decimal point should one round the neighbors distance to be considered equal neighbors. E.g., if round_decimals=1 and the two neighbors would be 3.44 and 3.36 Angstrom far, they will be considered equal. Defaults to None.
        """
        if magnetic_atoms: self.magnetic_atoms = magnetic_atoms
        if discard_nonmagnetic_atoms: self.discard_nonmagnetic_atoms = discard_nonmagnetic_atoms
        if magnetic_supercell: self.magnetic_supercell = magnetic_supercell
        if supercell_out_name: self.supercell_out_name = supercell_out_name
        if magnetic_moments: self.magnetic_moments = magnetic_moments
        if neighbor_cutoff: self.neighbor_cutoff = neighbor_cutoff
        if round_decimals: self.round_decimals = round_decimals
        if discard_nonmagnetic_atoms: self.remove_nonmagnetic_atoms()
        self.keep_only_first_unique_NN_label_combo = keep_only_first_unique_NN_label_combo
        print('self before making the supercell', self)
        self.make_supercell()
        print('self after making the supercell', self)
        if show_supercell: self.show_supercell_now()
        self.neighbors_analysis()
        if Heisenberg_Hamiltonian_type: self.define_Heisenberg_Hamiltonian(type=Heisenberg_Hamiltonian_type)

    def update_magnetic_moments(self, magnetic_moments):
        """Update magnetic moments before running 'get_total_energy()' method.

        Args:
            magnetic_moments (2D numpy array): see the 'initialize()' method for detailed explanation.
        """
        self.magnetic_moments = magnetic_moments
    
    def remove_nonmagnetic_atoms(self):
        """Remove nonmagnetic atoms from the structure.
        """
        self.remove_sites([i for i in range(len(self)) if i not in self.magnetic_atoms])

    def make_supercell(self):
        """Make a supercell of the uploaded unit cell and save it to a file. This supercell will serve as the magnetic unit cell.
        """
        super().make_supercell(self.magnetic_supercell)
        # label atoms by atom type + index
        # for i, site in enumerate(self):
        #     site.species = site.species_string + str(i+1)
        if self.supercell_out_name: self.to(self.supercell_out_name)
        
    def show_supercell_now(self):
        """Show the supercell in a new window using pymatgen's StructureVis.
        """
        # see https://pymatgen.org/pymatgen.vis.html#pymatgen.vis.structure_vtk.StructureVis
        # if you didn't remove nonmagnetic atoms, there will probably be multiple atom types
        print('Showing structure in a new window...\n')
        if not self.discard_nonmagnetic_atoms:
            # in that case ....
            raise Exception("Not working because visualizer throws error if multiple atom types present! Try setting 'discard_nonmagnetic_atoms=True'")
        vis_structure = Structure.from_file(self.supercell_out_name)
        visualizer = StructureVis()
        visualizer.set_structure(vis_structure)
        visualizer.show()

    def neighbors_analysis(self, order_by_NN_increasingly=True):
        """Analyze neighbors for each site in the magnetic unit cell.
            Calculate their distances from the given site, their order, and the number of neighbors of each order.
        
            Will be automatically run before the dependent methods: get_J1_with_phantoms_for_supercell(), get_total_energy()
        """
        # pymatgen's neighbor search:
                # find neighbors: neighbors is a list (for each site in the unit cell) of list of PeriodicNeighbor objects ... https://pymatgen.org/pymatgen.core.html#pymatgen.core.structure.PeriodicNeighbor
            # the return type is a [(site, distance) …]
        self.all_neighbors_for_all_sites = self.get_all_neighbors(self.neighbor_cutoff)

        # dimensions of neighbor matrices
        self.N_atoms_in_magnetic_unit_cell = len(self)
        self.N_neighbors_up_to_cutoff = len(self.all_neighbors_for_all_sites[0])
  
        # derived neighbor attributes
        self.all_neighbors_coords = self.get_neighbors_attribute('coords')
        self.all_neighbors_distances = self.get_all_neighbors_distances()
        self.all_neighbors_NN = np.array([nn_order_from_distances(self.all_neighbors_distances[i,:], round_decimals=self.round_decimals) for i in range(self.N_atoms_in_magnetic_unit_cell)])
        self.all_neighbors_NN_multiplicity = np.array([count_nn_order_neighbors(self.all_neighbors_NN[i,:]) for i in range(self.N_atoms_in_magnetic_unit_cell)])

        self.all_neighbors_labels = self.get_neighbors_attribute('species_string')

        # order neighbors by NN order
        if order_by_NN_increasingly:
            self.order_all_arrays_by_NN_increasingly()

        # create the multiplicity array: by default this is just an array of 1, unless they are grouped 
        #   as for instance in the  keep_only_first_unique_NNorder_label_combo_neighbor()  method where
        #   the multiplicity will be overwritten
        self.all_neighbors_NN_multiplicity = np.ones((self.N_atoms_in_magnetic_unit_cell, self.N_neighbors_up_to_cutoff), dtype=np.int32)

        # keep only first of the unique NN order neighbors
        if self.keep_only_first_unique_NN_label_combo:
            self.keep_only_first_unique_NNorder_label_combo_neighbor()

    def order_all_arrays_by_NN_increasingly(self):
        """Order all arrays by NN order.
        """
        # get ordering indeces
        # !!!!!! add secondary ordering by site label !!!!!
        self.all_neighbors_order = np.argsort(self.all_neighbors_NN, axis=1)

        # order all arrays (first is list of lists, rest are numpy arrays)
        self.all_neighbors_for_all_sites = [[row[ind] for ind in self.all_neighbors_order[i]] for i, row in enumerate(self.all_neighbors_for_all_sites)]
        self.all_neighbors_coords = np.take_along_axis(self.all_neighbors_coords, self.all_neighbors_order, axis=1)
        self.all_neighbors_distances = np.take_along_axis(self.all_neighbors_distances, self.all_neighbors_order, axis=1)
        self.all_neighbors_NN = np.take_along_axis(self.all_neighbors_NN, self.all_neighbors_order, axis=1)
        self.all_neighbors_labels = np.take_along_axis(self.all_neighbors_labels, self.all_neighbors_order, axis=1)

    def keep_only_first_unique_NNorder_label_combo_neighbor(self):
        """Keep only the first unique term of the given NN-site_label combination.
        """
        # copy the neighbors array to keep the original
        self.all_neighbors_for_all_sites_original = copy(self.all_neighbors_for_all_sites)
        # array of tuples of (order, label) for each site
        # [N_atoms_in_magnetic_unit_cell, N_neighbors] array of tuples (NN, label)
        self.all_neighbors_NNorder_label_combo = [[(self.all_neighbors_NN[i,j], self.all_neighbors_labels[i,j]) for j in range(self.N_neighbors_up_to_cutoff)] for i in range(self.N_atoms_in_magnetic_unit_cell)]

        # get unique NN orders and labels
        self.all_neighbors_NNorder_label_combo_unique_idx = np.array([np.unique(row, axis=0, return_index=True)[1] for row in self.all_neighbors_NNorder_label_combo])

        # get their number of appearances
        multiplicity = []
        len_character_tuple = len(self.all_neighbors_NNorder_label_combo[0][0])
        for i in range(self.N_atoms_in_magnetic_unit_cell):
            multiplicity.append([])
            for unique_combo_idx in self.all_neighbors_NNorder_label_combo_unique_idx[i,:]:
                mult_i = int(np.sum(np.sum(np.array(self.all_neighbors_NNorder_label_combo[i]) == self.all_neighbors_NNorder_label_combo[i][unique_combo_idx], axis=1) == len_character_tuple))
                multiplicity[i].append(mult_i)
        self.all_neighbors_NNorder_label_combo_unique_multiplicity = np.array(multiplicity)
        
        # keep only the first unique NN order neighbor
        self.all_neighbors_for_all_sites = [[self.all_neighbors_for_all_sites[i][j] for j in self.all_neighbors_NNorder_label_combo_unique_idx[i]] for i in range(self.N_atoms_in_magnetic_unit_cell)]
        self.all_neighbors_coords = np.array([self.all_neighbors_coords[i][self.all_neighbors_NNorder_label_combo_unique_idx[i]] for i in range(self.N_atoms_in_magnetic_unit_cell)])
        self.all_neighbors_distances = np.array([self.all_neighbors_distances[i][self.all_neighbors_NNorder_label_combo_unique_idx[i]] for i in range(self.N_atoms_in_magnetic_unit_cell)])
        self.all_neighbors_NN = np.array([self.all_neighbors_NN[i][self.all_neighbors_NNorder_label_combo_unique_idx[i]] for i in range(self.N_atoms_in_magnetic_unit_cell)])
        self.all_neighbors_labels = np.array([self.all_neighbors_labels[i][self.all_neighbors_NNorder_label_combo_unique_idx[i]] for i in range(self.N_atoms_in_magnetic_unit_cell)])

    def get_neighbors_attribute(self, attribute):
        """Generic method to get an array of attributes from the all_neighbors_for_all_sites array.

        Args:
            attribute (str): name of the attribute to get from the objects

        Returns:
            np.array: 2D array of attributes of all_neighbors_for_all_sites
        """
        array_of_attributes = np.zeros((self.N_atoms_in_magnetic_unit_cell, self.N_neighbors_up_to_cutoff), dtype=type(getattr(self.all_neighbors_for_all_sites[0][0][0], attribute)))
        for i, row in enumerate(self.all_neighbors_for_all_sites):
            for j, obj in enumerate(row):
                array_of_attributes[i,j] = getattr(obj, attribute)
        return array_of_attributes
    
    def get_all_neighbors_distances(self):
        """Get an array of distances from the all_neighbors_for_all_sites array.

        Returns:
            np.array: 2D array of distances of all_neighbors_for_all_sites
        """
        if not hasattr(self, 'all_neighbors_coords'):
            self.all_neighbors_coords = self.get_neighbors_attribute('coords')

        # for all the coords subtract the coords of the site, then run np.linalg.norm
        all_neighbors_distances = np.zeros((self.N_atoms_in_magnetic_unit_cell, self.N_neighbors_up_to_cutoff), dtype=np.float64)

        for i in range(self.N_atoms_in_magnetic_unit_cell):
            for j in range(self.N_neighbors_up_to_cutoff):
                all_neighbors_distances[i,j] = np.linalg.norm(self.all_neighbors_coords[i,j] - self[i].coords)
        return all_neighbors_distances

    def get_J1_with_phantoms_for_supercell(self, four_state_atoms_indeces=(0,1)):

        # run the neighbors_analysis() method if it hasn't been run yet
        if not hasattr(self, 'all_neighbors_for_all_sites'):
            self.neighbors_analysis()
                
        # print('All neighbors for all sites:', self.all_neighbors_for_all_sites)
        # print('len(all_neighbors_for_all_sites):', len(self.all_neighbors_for_all_sites))
        # print('len(all_neighbors_for_all_sites[0]):', len(self.all_neighbors_for_all_sites[0]))
        # print('type(all_neighbors_for_all_sites[0][0]):', type(self.all_neighbors_for_all_sites[0][0]))

        # for the first four-state-method atom find all his second-type four-state-method-atom friends
        #   : identify how many and which order nearest-neighbor atoms they are

        id1 = four_state_atoms_indeces[0]
        id2 = four_state_atoms_indeces[1]

        id1_coords = np.array(self[id1].coords)
        all_neighbors_for_id1 = self.all_neighbors_for_all_sites[id1]

        neighbors_of_id1_coords = np.array( [neighbor.coords for neighbor in all_neighbors_for_id1] )
        neighbors_of_id1_labels = [neighbor.species_string for neighbor in all_neighbors_for_id1]
        neighbors_of_id1_distances = np.linalg.norm(neighbors_of_id1_coords - id1_coords, axis=1)
        neighbors_of_id1_nn_order = nn_order_from_distances(neighbors_of_id1_distances, round_decimals=self.round_decimals)
        neighbors_of_id1_nn_number = count_nn_order_neighbors(neighbors_of_id1_nn_order)

        # neighbor_is_type_id2 = [neighbor.species_string == self[id2].species_string for neighbor in all_neighbors_for_id1]
        id2_coords = self[id2].coords
        neighbor_is_type_id2 = [np.allclose(neighbor.to_unit_cell().coords, id2_coords) for neighbor in all_neighbors_for_id1]
        
        neighbors_of_id1_of_type_id2_coords = neighbors_of_id1_coords[neighbor_is_type_id2]
        neighbors_of_id1_of_type_id2_distances = neighbors_of_id1_distances[neighbor_is_type_id2]
        neighbors_of_id1_of_type_id2_nn_order = neighbors_of_id1_nn_order[neighbor_is_type_id2]
        neighbors_of_id1_of_type_id2_nn_number = count_nn_order_neighbors(neighbors_of_id1_of_type_id2_nn_order)

        # # print('Distances to id1:', neighbors_of_id1_distances)
        # # print('Nearest neighbor orders:', neighbors_of_id1_nn_order)
        # print('Number of all neighbors of order for id1:', neighbors_of_id1_nn_number)
        # # print('Atom types: ', neighbors_of_id1_labels)

        # # print('Type-id2 nearest neighbor orders:', neighbors_of_id1_of_type_id2_nn_order)
        # print('Number of id2-type neighbors of order for id1:', neighbors_of_id1_of_type_id2_nn_number)
        # # list of indeces by neighbors

        # SAME for id2
        id2_coords = np.array(self[id2].coords)
        all_neighbors_for_id2 = self.all_neighbors_for_all_sites[id2]

        neighbors_of_id2_coords = np.array( [neighbor.coords for neighbor in all_neighbors_for_id2] )
        neighbors_of_id2_labels = [neighbor.species_string for neighbor in all_neighbors_for_id2]
        neighbors_of_id2_distances = np.linalg.norm(neighbors_of_id2_coords - id2_coords, axis=1)
        neighbors_of_id2_nn_order = nn_order_from_distances(neighbors_of_id2_distances, round_decimals=self.round_decimals)
        neighbors_of_id2_nn_number = count_nn_order_neighbors(neighbors_of_id2_nn_order)

        # neighbor_is_type_id1 = [neighbor.species_string == self[id1].species_string for neighbor in all_neighbors_for_id2]
        id1_coords = self[id1].coords
        neighbor_is_type_id1 = [np.allclose(neighbor.to_unit_cell().coords, id1_coords) for neighbor in all_neighbors_for_id2]
        print('neighbor is type id1', neighbor_is_type_id1)
        neighbors_of_id2_of_type_id1_coords = neighbors_of_id2_coords[neighbor_is_type_id1]
        neighbors_of_id2_of_type_id1_distances = neighbors_of_id2_distances[neighbor_is_type_id1]
        neighbors_of_id2_of_type_id1_nn_order = neighbors_of_id2_nn_order[neighbor_is_type_id1]
        neighbors_of_id2_of_type_id1_nn_number = count_nn_order_neighbors(neighbors_of_id2_of_type_id1_nn_order)

        # # print('Distances to id2:', neighbors_of_id2_distances)
        # # print('Nearest neighbor orders:', neighbors_of_id2_nn_order)
        # print('Number of all neighbors of order for id2:', neighbors_of_id2_nn_number)
        # # print('Atom types: ', neighbors_of_id2_labels)

        # # print('Type-id1 nearest neighbor orders:', neighbors_of_id2_of_type_id1_nn_order)
        # print('Number of id1-type neighbors of order for id2:', neighbors_of_id2_of_type_id1_nn_number)

        # sum them up
        J1_with_phantoms = np.array(neighbors_of_id1_of_type_id2_nn_number) + np.array(neighbors_of_id2_of_type_id1_nn_number)
        # the lowest -nearest neighbor interaction incorrectly includes one more interaction
        # lowest non-zero value
        # index for lowest non-zero value of J1_with_phantoms
        for i in range(len(J1_with_phantoms)):
            if J1_with_phantoms[i] != 0:
                J1_with_phantoms[i] -= 1
                break    

        # print('J1 with phantoms:', J1_with_phantoms)
        neighbors_of_id1_distances_unique = np.sort(np.unique(neighbors_of_id1_distances.round(decimals=self.round_decimals)))
        
        # save the calculated data as object attributes
        self.id1_coords = id1_coords
        self.neighbors_of_id1_coords = neighbors_of_id1_coords
        self.neighbors_of_id1_nn_order = neighbors_of_id1_nn_order
        self.neighbors_of_id1_distances = neighbors_of_id1_distances
        self.neighbors_of_id1_of_type_id2_coords = neighbors_of_id1_of_type_id2_coords
        self.id2_coords = id2_coords
        self.neighbors_of_id2_coords = neighbors_of_id2_coords
        self.neighbors_of_id2_nn_order = neighbors_of_id2_nn_order
        self.neighbors_of_id2_distances = neighbors_of_id2_distances
        self.neighbors_of_id2_of_type_id1_coords = neighbors_of_id2_of_type_id1_coords

        self.J1_with_phantoms = J1_with_phantoms
        self.neighbors_of_id1_distances_unique = neighbors_of_id1_distances_unique


    def plot_J1_with_phantoms(self):
        J1_with_phantoms_plot = copy(self.J1_with_phantoms).astype(float)

        # so that zero values are still visible
        J1_with_phantoms_plot[J1_with_phantoms_plot == 0] = 0.1

        fig, ax = plt.subplots(figsize=(4.5,3))
        ylim = 8
        plt.bar(self.neighbors_of_id1_distances_unique[:len(self.J1_with_phantoms)], J1_with_phantoms_plot, \
                    width=0.15, align='center', color='k')

        plt.xlabel('Distance (Angstroms)')
        plt.ylabel('Number of interactions')
        plt.title(r'$J_1$' + ' with phantoms for supercell ' + str(self.magnetic_supercell))
        # plt.ylim(0, ylim)
        plt.xlim(0, self.neighbor_cutoff*1.05)
        # y ticks in steps of 2
        # plt.yticks(np.arange(ax.get_ylim()[0], ax.get_ylim()[1], 2))
        # grid parallel to x-axis with step of 1 between lines
        plt.grid(axis='y', linestyle='-', linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_four_state_neighbors(self, invert_colors=False, title='4-state method'):
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.set_aspect('equal')
        ax.set_xlim(-self.neighbor_cutoff*1.4, self.neighbor_cutoff*1.4)
        ax.set_ylim(-self.neighbor_cutoff*1.4, self.neighbor_cutoff*1.4)
        ax.set_xlabel(r'$x$ ($\mathrm{\AA}$)')
        ax.set_ylabel(r'$y$ ($\mathrm{\AA}$)')
        ax.set_title(title)
        # radii circles
        for r in np.unique(self.neighbors_of_id1_distances.round(decimals=self.round_decimals)):
            ax.add_patch(Circle((self.id1_coords[0], self.id1_coords[1]), radius=r, fill=False, color='k', linewidth=0.25))
        # atoms
        size = 100
        ax.scatter(self.neighbors_of_id1_coords[:,0], self.neighbors_of_id1_coords[:,1], facecolors='w', edgecolors='w', linewidths=1, s=size*1.35, label='neighbor_margin')
        ax.scatter(self.neighbors_of_id1_coords[:,0], self.neighbors_of_id1_coords[:,1], facecolors='w', edgecolors='k', linewidths=1, s=size, label='neighbors')
        for i, xy in enumerate(self.neighbors_of_id1_coords):
            ax.annotate(self.neighbors_of_id1_nn_order[i], (xy[0], xy[1]), fontsize=6, color='k', ha='center', va='center')
        # four-state atoms
        c1 = 'r' if invert_colors else 'g'
        c2 = 'g' if invert_colors else 'r'
        ax.scatter(self.id1_coords[0], self.id1_coords[1], facecolors='w', edgecolors=c1, linewidths=1, s=size, label='id1')
        ax.annotate('0', (self.id1_coords[0], self.id1_coords[1]), fontsize=6, color='k', ha='center', va='center')

        for id2_each_coords in self.neighbors_of_id1_of_type_id2_coords:
            ax.scatter(id2_each_coords[0], id2_each_coords[1], facecolors='w', edgecolors=c2, linewidths=1, s=size, label='id2')
        # ax.annotate(neighbors_of_id1_nn_order[id2], (id2_coords[0], id2_coords[1]), fontsize=6, color='k', ha='center', va='center')

        # plot structure's unit cell
        unit_cell_2D_vectors = self.lattice.matrix[:2, :2]
        polygon = [[0,0], unit_cell_2D_vectors[0], unit_cell_2D_vectors[0] + unit_cell_2D_vectors[1], unit_cell_2D_vectors[1]]
        ax.add_patch(Polygon(polygon, fill=False, color='k', linewidth=1.0))
        
        # print('neighbors_of_id1_coords\n', self.neighbors_of_id1_coords)
        # print('neighbors_of_id1_nn_order\n', self.neighbors_of_id1_nn_order)
        plt.show()
    
    def define_Heisenberg_Hamiltonian(self, type='isotropic'):
        """Heisenberg Hamiltonian between two spins

        Args:
            type (str, optional): Type of Hamiltonian. Defaults to 'isotropic'.
        """
        self.HH = HeisenbergHamiltonian(type=type)

    def get_total_energy(self):
        """

                1. create the arrays site_spins, site_labels, and site_multiplicity needed by 
                    HeisenbergHamiltonian.whole_system_energy()
                2. then run HeisenbergHamiltonian.whole_system_energy()
                3. flatten the output array into a 1D array in a reasonable (user-defined?) way
        """

        # --- 1. create the arrays ----
            # dimensions of the 2D arrays: (N_sites_magnetic_unit_cell, N_neighbors_up_to_cutoff)
            #   - same as all_neighbors_for_all_sites

        if not hasattr(self, 'all_neighbors_for_all_sites'):
            self.neighbors_analysis()

        # make an array of magnetic moments for all neighbors of all atoms
        self.all_neighbors_spins = np.array([[self.magnetic_moments[self.all_neighbors_for_all_sites[i][j].index] for j in range(len(self.all_neighbors_for_all_sites[i]))] for i in range(self.N_atoms_in_magnetic_unit_cell)])

        # --- 2. run HeisenbergHamiltonian.whole_system_energy() ----
        self.HH.get_total_energy(self.magnetic_moments, self.all_neighbors_spins, self.all_neighbors_labels, self.all_neighbors_NNorder_label_combo_unique_multiplicity)

        # --- 3. flatten the output array ----
           # group depending on the NN order and site labels
           # order depending on the required order
           #    most sensible (interaction_type, label1-label2, NN_order), 
           #       e.g. (Jxx_Cr1Cr2_NN1, Jxx_Cr1Cr2_NN2, Jxx_Cr1Cr3_NN1, Jxx_Cr1Cr3_NN2, Jzz_Cr1Cr2_NN1, Jzz_Cr1Cr2_NN2, Jzz_Cr1Cr3_NN1, Jzz_Cr1Cr3_NN2)
           #    create labels for this order
        
        print('two_site_parameters', self.HH.two_site_parameters)
        # print('two_site_prefactors:', self.HH.two_site_prefactors)

        # create an array of label pairs

        # -> order by the label pairs

        # now everything is ordered (NN order, label pairs, interaction type as a list)
        #  -> flatten the array: let user decide the order of indeces

