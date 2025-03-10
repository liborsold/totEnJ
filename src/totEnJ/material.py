from pymatgen.core import Lattice, Structure, Molecule, PeriodicNeighbor, Composition
from pymatgen.symmetry.groups import PointGroup
from pymatgen.symmetry.analyzer import PointGroupAnalyzer, cluster_sites, SpacegroupAnalyzer
from pymatgen.vis.structure_vtk import StructureVis
from os.path import exists
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from matplotlib.patches import Circle, Polygon
from totEnJ.utils import nn_order_from_distances, count_nn_order_neighbors, leave_only_first_part_with_letters
from totEnJ.HeisenbergHamiltonian import HeisenbergHamiltonian
import pandas as pd
import warnings

class StructureJ(Structure):
    """Class for the magnetic structure analysis and calculation of the total energy using the Heisenberg Hamiltonian, derived from pymatgen's Structure.
    """
    
    def initialize(self, magnetic_atoms=None, discard_nonmagnetic_atoms=None, magnetic_supercell=None, 
                   show_supercell=None, supercell_out_name=None, Heisenberg_Hamiltonian_type=None, 
                   magnetic_moments=None, neighbor_cutoff=None, round_decimals=None, symprec_group_analyzer=0.01, 
                   verbose=False):
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
        self.symprec_group_analyzer = symprec_group_analyzer
        
        if magnetic_atoms: self.magnetic_atoms = magnetic_atoms
        if discard_nonmagnetic_atoms: self.discard_nonmagnetic_atoms = discard_nonmagnetic_atoms
        if magnetic_supercell: self.magnetic_supercell = magnetic_supercell
        if supercell_out_name: self.supercell_out_name = supercell_out_name
        if magnetic_moments: self.magnetic_moments = magnetic_moments
        if neighbor_cutoff: self.neighbor_cutoff = neighbor_cutoff
        if round_decimals: self.round_decimals = round_decimals
        # symmetry analysis must be done before removing nonmagnetic atoms - they influence the symmetry in any case!!
        self.symmetry_analysis()
        if discard_nonmagnetic_atoms: self.remove_nonmagnetic_atoms()
        if Heisenberg_Hamiltonian_type: self.define_Heisenberg_Hamiltonian(type=Heisenberg_Hamiltonian_type)
        if verbose: print('self before making the supercell', self)
        self.make_supercell()
        if verbose: print('self after making the supercell', self)
        if show_supercell: self.show_supercell_now()

        print('NUMBER OF SITES IN SYSTEM', len(self))

        self.neighbors_analysis()

        self.neighbors_analysis_clustering_custom_made()

        self.do_pandas_magic()

    def symmetry_analysis(self):
        # get space group and point group
        sga = SpacegroupAnalyzer(self, symprec=self.symprec_group_analyzer)
        # save space group symbol and point group symbol, also point group object
        self.sg_symbol = sga.get_space_group_symbol()
        self.pg_symbol =  sga.get_point_group_symbol()
        self.pg = PointGroup(self.pg_symbol)  

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

    def masked_array_from_list_of_lists(self, list_of_lists, mask=None):
        """Create a masked array from a list of lists with inhomogeneous dimensions of the 1D lists.

        Args:
            list_of_lists (list of lists): List of lists to convert to a masked array.

        Returns:
            numpy.ma.masked_array: Masked array.
        """
        # length of the longest list
        max_len = max([len(l) for l in list_of_lists])
        # pad all lists to the same length
        list_of_lists = [l + [np.nan]*(max_len-len(l)) for l in list_of_lists]
        # create mask automatically if it was not provided (better to provide a single mask to all calls)
        if mask is None:
            mask = self.all_neighbors_mask
        return np.ma.masked_array(list_of_lists, mask=mask)

    def neighbors_analysis(self, verbose=False):
        """Analyze neighbors for each site in the magnetic unit cell.
            Calculate their distances from the given site, their order, and the number of neighbors of each order.
        
            Will be automatically called before the dependent methods: get_J1_with_phantoms_for_supercell(), get_total_energy()
        """
        # pymatgen's neighbor search:
        # find neighbors: neighbors is a list (for each site in the unit cell) of list of PeriodicNeighbor objects ... https://pymatgen.org/pymatgen.core.html#pymatgen.core.structure.PeriodicNeighbor
            # the return type is a [(site, distance) …]
        self.all_neighbors_for_all_sites = self.get_all_neighbors(self.neighbor_cutoff)

        # number of neighbors for each site
        self.all_neighbors_N_neighbors = []
        for neighbors in self.all_neighbors_for_all_sites:
            self.all_neighbors_N_neighbors.append(len(neighbors))
        if verbose: print('number of neighbors for each site:', self.all_neighbors_N_neighbors)

        # dimensions of neighbor matrices
        self.N_atoms_in_magnetic_unit_cell = len(self)
        self.N_neighbors_up_to_cutoff = max(self.all_neighbors_N_neighbors)
  
        # ==== DERIVED ATTRIBUTES ====

        # -- mask -- for the 'masked 2D numpy arrays' with attributes for all neighbors of all sites
        #    e.g. if all_neighbors_N_neighbors = [3, 4, 2], then the mask will be [[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0]]
        #    to ensure that only the data for really existing are considered
        self.all_neighbors_mask = np.zeros((self.N_atoms_in_magnetic_unit_cell, self.N_neighbors_up_to_cutoff), dtype=bool)
        for i, N_neighbors in enumerate(self.all_neighbors_N_neighbors):
            self.all_neighbors_mask[i,:N_neighbors] = True

        self.all_neighbors_coords = self.get_neighbors_coords()
        for i in range(self.N_atoms_in_magnetic_unit_cell):
            print('self.all_neighbors_coords[i]:', self.all_neighbors_coords[i])
            print('self[i].coords', self[i].coords)
            # exit()
        # self.all_neighbors_rij = np.array([coords - self[i].coords for i in range(self.N_atoms_in_magnetic_unit_cell) for coords in self.all_neighbors_coords[i]])
        self.all_neighbors_rij = np.zeros((self.N_atoms_in_magnetic_unit_cell, self.N_neighbors_up_to_cutoff, 3), dtype=np.ndarray)
        for i in range(self.N_atoms_in_magnetic_unit_cell):
            for j, coords in enumerate(self.all_neighbors_coords[i]):
                print('i, j', i, j)
                print('fuck this shit', coords - self[i].coords)
                self.all_neighbors_rij[i,j] = coords - self[i].coords

                
        # print('self.all_neighbors_rij', self.all_neighbors_rij)
        print('good')
        self.all_neighbors_distances = self.get_all_neighbors_distances()
        print('still good')
        self.all_neighbors_NN = np.array([nn_order_from_distances(self.all_neighbors_distances[i,:], round_decimals=self.round_decimals) for i in range(self.N_atoms_in_magnetic_unit_cell)])

        if verbose:
            print('all_neighbors_NN:', self.all_neighbors_NN)

        self.center_atom_labels = self.get_center_atom_attribute('species_string')
        self.all_neighbors_labels = self.get_neighbors_attribute('species_string')

        self.all_neighbors_images = self.get_neighbors_attribute('image')

        self.center_atom_index = self.get_center_atom_index()
        self.all_neighbors_index = self.get_neighbors_attribute('index')

    def get_neighbors_coords(self):
        """Get an array of coordinates from the all_neighbors_for_all_sites array.

        Returns:
            list of numpy ndarrays: for each site in the magnetic unit cell, an array (N_neighbors, 3) of coordinates of neighbors
        """
        coords_neat = []
        # print("self.get_neighbors_attribute('coords')", self.get_neighbors_attribute('coords'))
        # exit()
        for i, coords_list in enumerate(self.get_neighbors_attribute('coords')):
            N_neighbors = self.all_neighbors_N_neighbors[i]
            try:
                coords_list_np_ndarray = np.stack(coords_list[N_neighbors-self.N_neighbors_up_to_cutoff:], axis=0)
            except ValueError:
                raise Exception("Some of the neighbors (furhest ones probably) give '.coords' as 0.\n\n============>    TRY INCREASING THE NEIGHBOR_CUTOFF DISTANCE FROM THE VERY BEGINNING     <============\n\nAborting.")

            coords_neat.append(coords_list_np_ndarray)
        return coords_neat

    def get_cluster_index_for_all(self):
        """Create an array of type 'all_neighbors' with the cluster index for each neighbor.
            The cluster index is ascending (1-based) and grouped by chemical element (i.e., two 
                    neighbors of the same atom will can have the same index if they have a different
                    site label.

            Saves:
                self.all_neighbors_cluster_index (np.array): 2D array of cluster indeces for each neighbor for all atoms
        """
        # cluster index for all neighbors
        self.all_neighbors_cluster_index = np.zeros((self.N_atoms_in_magnetic_unit_cell, self.N_neighbors_up_to_cutoff), dtype=np.int32)
        # sweep over clusters
        for i in range(self.N_atoms_in_magnetic_unit_cell):
            for i_cluster, cluster in enumerate(self.all_clusters[i]):
                cluster_idx = self.all_clusters_idx_by_dist_chem_grouped[i][i_cluster]
                for j in cluster:
                    self.all_neighbors_cluster_index[i,j] = cluster_idx

    def do_pandas_magic(self, verbose=False):
        # ==== PANDAS TABLE OF TWO-SITE INTERACTIONS ====
        self.two_site_interaction_table = pd.DataFrame()
        data = [array2D.flatten() for array2D in [self.center_atom_index, self.center_atom_labels, self.all_neighbors_index, self.all_neighbors_labels, self.all_neighbors_images, self.all_neighbors_distances, self.all_neighbors_NN, self.all_neighbors_cluster_index]]
        column_names = ['center_atom_index', 'center_atom_label', 'neighbor_index', 'neighbor_label', 'neighbor_image', 'distance', 'NN_order', 'cluster_index']
        for i, name in enumerate(column_names):
            self.two_site_interaction_table[name] = data[i]
            
        all_neighbors_rij_list = [self.all_neighbors_rij[i][j] for i in range(len(self.all_neighbors_rij)) for j in range(len(self.all_neighbors_rij[i]))]
        self.two_site_interaction_table['r_ij'] = all_neighbors_rij_list
        self.get_total_energy_prefactors_for_pandas_table()

        self.aggregate_pandas_table()

        if verbose:
            print('Two-site interaction table:\n', self.two_site_interaction_table)
            print('------------ data:', self.center_atom_labels[0,0])

    # !!!!! DO NOT USE THE BELOW - IF YOU FORGET TO ORDER SOME ARRAYS THERE WILL BE HARD-TO-FIND BUGS !!!!!
    # def order_all_arrays_by_NN_increasingly(self):
    #     """Order all arrays by NN order.
    #     """
    #     # get ordering indeces
    #     # !!!!!! add secondary ordering by site label !!!!!
    #     self.all_neighbors_order = np.argsort(self.all_neighbors_NN, axis=1)

    #     # order all arrays (first is list of lists, rest are numpy arrays)
    #     self.all_neighbors_for_all_sites = [[row[ind] for ind in self.all_neighbors_order[i]] for i, row in enumerate(self.all_neighbors_for_all_sites)]
    #     self.all_neighbors_coords = np.take_along_axis(self.all_neighbors_coords, self.all_neighbors_order, axis=1)
    #     self.all_neighbors_distances = np.take_along_axis(self.all_neighbors_distances, self.all_neighbors_order, axis=1)
    #     self.all_neighbors_NN = np.take_along_axis(self.all_neighbors_NN, self.all_neighbors_order, axis=1)
    #     self.all_neighbors_labels = np.take_along_axis(self.all_neighbors_labels, self.all_neighbors_order, axis=1)

    def get_neighbors_attribute(self, attribute):
        """Generic method to get an array of attributes from the all_neighbors_for_all_sites array.

        Args:
            attribute (str): name of the attribute to get from the objects

        Returns:
            np.ma.masked_array: 2D array of attributes of all_neighbors_for_all_sites masked by 
                self.all_neighbors_mask (to ensure that only the data for really existing neighbors are considered)
        """
        typ_example = getattr(self.all_neighbors_for_all_sites[0][0][0], attribute)
        typ = type(typ_example)
        if typ == np.ndarray:
            typ_example.fill(-999   )
        print('typ', typ)
        if typ == str:
            typ = 'U8'
        data = np.zeros((self.N_atoms_in_magnetic_unit_cell, self.N_neighbors_up_to_cutoff), dtype=typ)
        data.fill(typ_example)
        # print('data', data)
        array_of_attributes = np.ma.masked_array(data, 
                                                    mask=self.all_neighbors_mask)
        for i, row in enumerate(self.all_neighbors_for_all_sites):
            for j, obj in enumerate(row):
                array_of_attributes[i,j] = getattr(obj, attribute)
        print('array_of_attributes', array_of_attributes[0,:-6])
        return array_of_attributes
    
    def get_center_atom_attribute(self, attribute):
        """Generic method to get an array of attributes for the center atom of all the neighbors.
        Has the same dimensions as 'get_neighbors_attribute()' result.

        Args:
            attribute (str): name of the attribute to get from the center atom

        Returns:
            np.array: 2D array of attributes of all_neighbors_for_all_sites
        """
        typ = type(getattr(self.all_neighbors_for_all_sites[0][0][0], attribute))
        # if typ is string-like
        if typ == str:
            typ = 'U8'
        array_of_attributes = np.ma.masked_array(np.zeros((self.N_atoms_in_magnetic_unit_cell, self.N_neighbors_up_to_cutoff), dtype=typ),
                                                    mask=self.all_neighbors_mask)
        for i, row in enumerate(self.all_neighbors_for_all_sites):
            for j, obj in enumerate(row):
                # for all j it is identical, because here we only care about i - the center atom
                array_of_attributes[i,j] = getattr(self[i], attribute)
        return array_of_attributes        
    
    def get_center_atom_index(self):
        """Get the index of the center atom for each neighbor.

        Returns:
            np.array: 2D array of center atom indeces
        """
        return np.array([[i for j in range(self.N_neighbors_up_to_cutoff)] for i in range(self.N_atoms_in_magnetic_unit_cell)])
    
    def get_all_neighbors_distances(self):
        """Get an array of distances from the all_neighbors_for_all_sites array.

        Returns:
            np.ma.masked_array: 2D array of distances of all_neighbors_for_all_sites (masked by self.all_neighbors_mask)
        """
        if not hasattr(self, 'all_neighbors_coords'):
            self.all_neighbors_coords = self.get_neighbors_coords()

        # for all the coords subtract the coords of the site, then run np.linalg.norm
        all_neighbors_distances = np.ma.masked_array(np.zeros((self.N_atoms_in_magnetic_unit_cell, self.N_neighbors_up_to_cutoff), dtype=np.float64),
                                                    mask=self.all_neighbors_mask)
        for i in range(self.N_atoms_in_magnetic_unit_cell):
            for j in range(self.N_neighbors_up_to_cutoff):
                all_neighbors_distances[i,j] = np.linalg.norm(self.all_neighbors_rij[i,j])
        return all_neighbors_distances

    def get_J1_with_phantoms_for_supercell(self, four_state_atoms_indeces=(0,1)):
        # run the neighbors_analysis() method if it hasn't been run yet
        if not hasattr(self, 'all_neighbors_for_all_sites'):
            # !!! YOU WILL PROBABLY HAVE TO SWITCH OFF KEEPING THE FIRST UNIQUE NN LABEL COMBO !!!
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

        neighbors_of_id1_coords = self.all_neighbors_coords[id1]
        neighbors_of_id1_labels = self.all_neighbors_labels[id1]
        neighbors_of_id1_distances = self.all_neighbors_distances[id1]
        neighbors_of_id1_nn_order = self.all_neighbors_cluster_index[id1]
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

        neighbors_of_id2_coords = np.array(self.all_neighbors_coords[id2])
        neighbors_of_id2_labels = self.all_neighbors_labels[id2]
        neighbors_of_id2_distances = self.all_neighbors_distances[id2]
        neighbors_of_id2_nn_order = self.all_neighbors_cluster_index[id2]
        
        # neighbor_is_type_id1 = [neighbor.species_string == self[id1].species_string for neighbor in all_neighbors_for_id2]
        id1_coords = self[id1].coords
        neighbor_is_type_id1 = [np.allclose(neighbor.to_unit_cell().coords, id1_coords) for neighbor in all_neighbors_for_id2]
        # print('neighbor is type id1', neighbor_is_type_id1)
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
        # # NO IT DOES NOT!  the lowest -nearest neighbor interaction incorrectly includes one more interaction
        # # lowest non-zero value
        # # index for lowest non-zero value of J1_with_phantoms
        # for i in range(len(J1_with_phantoms)):
        #     if J1_with_phantoms[i] != 0:
        #         J1_with_phantoms[i] -= 1
        #         break    

        print('J1 with phantoms:', J1_with_phantoms)
        neighbors_of_id1_distances_unique = np.sort(np.unique(neighbors_of_id1_distances.round(decimals=self.round_decimals)))
        
        # save the calculated data as object attributes
        self.four_state_atoms_indeces = four_state_atoms_indeces
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

    def plot_four_state_neighbors(self, invert_id1_and_id2=False, title='4-state method'):
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.set_aspect('equal')
        ax.set_xlim(-self.neighbor_cutoff*1.4, self.neighbor_cutoff*1.4)
        ax.set_ylim(-self.neighbor_cutoff*1.4, self.neighbor_cutoff*1.4)
        ax.set_xlabel(r'$x$ ($\mathrm{\AA}$)')
        ax.set_ylabel(r'$y$ ($\mathrm{\AA}$)')
        ax.set_title(title)

        if invert_id1_and_id2:
            neighbors_of_id_distances = self.neighbors_of_id2_distances
            id_coords = self.id2_coords
            neighbors_of_id_coords = self.neighbors_of_id2_coords
            neighbors_of_id_nn_order = self.neighbors_of_id2_nn_order
            neighbors_of_idx_of_type_idy_coords = self.neighbors_of_id2_of_type_id1_coords
        else:
            neighbors_of_id_distances = self.neighbors_of_id1_distances
            id_coords = self.id1_coords
            neighbors_of_id_coords = self.neighbors_of_id1_coords
            neighbors_of_id_nn_order = self.neighbors_of_id1_nn_order
            neighbors_of_idx_of_type_idy_coords = self.neighbors_of_id1_of_type_id2_coords

        c1 = 'r' if invert_id1_and_id2 else 'g'
        c2 = 'g' if invert_id1_and_id2 else 'r'

        # radii circles
        for r in np.unique(neighbors_of_id_distances.round(decimals=self.round_decimals)):
            ax.add_patch(Circle((id_coords[0], id_coords[1]), radius=r, fill=False, color='k', linewidth=0.25))
        # atoms
        size = 100
        ax.scatter(neighbors_of_id_coords[:,0], neighbors_of_id_coords[:,1], facecolors='w', edgecolors='w', linewidths=1, s=size*1.35, label='neighbor_margin')
        ax.scatter(neighbors_of_id_coords[:,0], neighbors_of_id_coords[:,1], facecolors='w', edgecolors='k', linewidths=1, s=size, label='neighbors')
        for i, xy in enumerate(neighbors_of_id_coords):
            ax.annotate(neighbors_of_id_nn_order[i], (xy[0], xy[1]), fontsize=6, color='k', ha='center', va='center')
        # four-state atoms

        ax.scatter(id_coords[0], id_coords[1], facecolors='w', edgecolors=c1, linewidths=1, s=size, label='id1')
        ax.annotate('0', (id_coords[0], id_coords[1]), fontsize=6, color='k', ha='center', va='center')

        for id_each_coords in neighbors_of_idx_of_type_idy_coords:
            ax.scatter(id_each_coords[0], id_each_coords[1], facecolors='w', edgecolors=c2, linewidths=1, s=size, label='id2')
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

    def get_all_neighbors_multiplicity(self):
        """Get an array in the form of 'all_neighbors_NN_multiplicity' but here it will be just 1 for all the 
            neighbors. 
        """
        all_neighbors_multiplicity = np.ones((self.N_atoms_in_magnetic_unit_cell, self.N_neighbors_up_to_cutoff), dtype=np.float64)
        self.all_neighbors_multiplicity = all_neighbors_multiplicity

    def get_total_energy(self, verbose=False):
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
        
        # !!!!!!!!!
        if not hasattr(self, 'all_neighbors_multiplicity'):
            self.get_all_neighbors_multiplicity()
        # !!!!!!!!!
        
        self.HH.get_total_energy(self.magnetic_moments, self.all_neighbors_spins, self.all_neighbors_labels, self.all_neighbors_multiplicity)

        # --- 3. flatten the output array ----
           # group depending on the NN order and site labels
           # order depending on the required order
           #    most sensible (interaction_type, label1-label2, NN_order), 
           #       e.g. (Jxx_Cr1Cr2_NN1, Jxx_Cr1Cr2_NN2, Jxx_Cr1Cr3_NN1, Jxx_Cr1Cr3_NN2, Jzz_Cr1Cr2_NN1, Jzz_Cr1Cr2_NN2, Jzz_Cr1Cr3_NN1, Jzz_Cr1Cr3_NN2)
           #    create labels for this order


           # A. START SLOOOOOW
            # --- first do the summation by order for a single parameter - e.g., the J
        prefactors = self.HH.two_site_prefactors[:,:,0].flatten()
        orders = self.all_neighbors_NN.flatten()

        prefactors_summed = []
        for i in range(1, max(orders)+1):
            prefactors_summed.append(np.sum(prefactors[orders == i]))
        
        # prepend total energy of the unit cell times number of unit cells in the supercell
        prefactors_summed.insert(0, np.prod(self.magnetic_supercell))

        # list to numpy array
        prefactors_summed = np.array(prefactors_summed)
        
        if verbose: print('single_site_parameters', self.HH.single_site_parameters)
        if verbose: print('two_site_parameters', self.HH.two_site_parameters)
        
        self.prefactors_summed = prefactors_summed
        # print('two_site_prefactors:', self.HH.two_site_prefactors)

        # create an array of label pairs

        # -> order by the label pairs

        # now everything is ordered (NN order, label pairs, interaction type as a list)
        #  -> flatten the array: let user decide the order of indeces

    def get_total_energy_clustering(self):
        """Get the total energy using unique sites and clustered neighbors.
        """
        # make a pandas array and fill it with all the pre-computed arrays


    def plot_system_spin_config(self, id_main_atom=0):
        """Plot the system atoms with arrows for spin for the current spin configuration.
        This is different from plotting four-state configurations, where we only care about colouring two chosen atoms and counting number of their interactions.
        Here it is the whole spin configuration that matters.
        These configurations are used for diagonalization.
        """
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.set_aspect('equal')
        ax.set_xlim(-self.neighbor_cutoff*1.4, self.neighbor_cutoff*1.4)
        ax.set_ylim(-self.neighbor_cutoff*1.4, self.neighbor_cutoff*1.4)
        ax.set_xlabel(r'$x$ ($\mathrm{\AA}$)')
        ax.set_ylabel(r'$y$ ($\mathrm{\AA}$)')
        ax.set_title('Spin configuration')

        id_coords = self[id_main_atom].coords
        neighbors_of_id_coords = self.all_neighbors_coords[id_main_atom]
        print('type', type(neighbors_of_id_coords))
        print('shape', neighbors_of_id_coords.shape)
        print('neighbors_of_id_coords\n', neighbors_of_id_coords)

        neighbors_of_id_nn_order = self.all_neighbors_NN[id_main_atom]

        # radii circles
        for r in np.unique(self.all_neighbors_distances[id_main_atom].round(decimals=self.round_decimals)):
            ax.add_patch(Circle((id_coords[0], id_coords[1]), radius=r, fill=False, color='k', linewidth=0.25))
        # atoms
        size = 100
        ax.scatter(neighbors_of_id_coords[:,0], neighbors_of_id_coords[:,1], facecolors='w', edgecolors='w', linewidths=1, s=size*1.35, label='neighbor_margin')
        ax.scatter(neighbors_of_id_coords[:,0], neighbors_of_id_coords[:,1], facecolors='w', edgecolors='k', linewidths=1, s=size, label='neighbors')
        for i, xy in enumerate(neighbors_of_id_coords):
            ax.annotate(neighbors_of_id_nn_order[i], (xy[0], xy[1]), fontsize=6, color='k', ha='center', va='center')
        
        ax.scatter(id_coords[0], id_coords[1], facecolors='w', edgecolors='k', linewidths=1, s=size, label='id1')
        ax.annotate('0', (id_coords[0], id_coords[1]), fontsize=6, color='k', ha='center', va='center')

        # plot structure's unit cell
        unit_cell_2D_vectors = self.lattice.matrix[:2, :2]
        polygon = [[0,0], unit_cell_2D_vectors[0], unit_cell_2D_vectors[0] + unit_cell_2D_vectors[1], unit_cell_2D_vectors[1]]
        ax.add_patch(Polygon(polygon, fill=False, color='k', linewidth=1.0))
        
        plt.show()


    def neighbors_analysis_clustering(self, verbose=True):
        """Find unique sites of the symmetrized magnetic unit cell.
        Label by unique labels! Attribute unique lables (chemical element + integer if multiple non-equivalent chemical element sites present).
        For all atoms perform a clustering analysis.
        Create a mapping of site_i-site_j to interaction identifier:
            - site: ((periodic image tuple), site index)
            - interaction identifier in the form (label_i, label_j, NN_order)

        Args:
            verbose (bool, optional): Print the results of the clustering analysis. Defaults to True.

        Saves:
            self.equiv_site_idx (list of lists): Each list contains the indeces of the equivalent sites.
            self.N_unique_sites_in_unitcell (int): Number of unique sites in the unit cell.
            self.neighbors_of_each_first_equiv_site (list of dicts): Each dict contains the clustering of the neighbors of the first site of the equivalent sites.
        """

        self.create_labels_for_unique_sites()

        if verbose:
            print('self.equiv_site_idx', self.equiv_site_idx)
            print('SYTEM LABELED by unique sites', self)

        self.all_clusters = []

        # ==== 2. CLUSTERs of neighbors for all sites ====
        for i in range(len(self)):

            # get neighbors for the first site of the equivalent sites
            neighbors_i = self.all_neighbors_for_all_sites[i]

            # get the point group of neighbors of atom i
            molecule_i = Molecule.from_sites(neighbors_i)
            pga = PointGroupAnalyzer(molecule_i)

            if verbose: 
                print(pga.sch_symbol) # show the point group
                print(molecule_i.center_of_mass) # show the point group's center - should be very close to self[i].coords

            # cluster atoms
            cluster_i = cluster_sites(molecule_i, 10**(-self.round_decimals))

            self.all_clusters.append(cluster_i[1])

            # cluster_i[0] should give the origin site (atom i) or None, if there is no atom there
            # #   gives None in our case
            if verbose: 
                # cluster_i[1] gives a dict of {(avg_dist, species_and_occu): [list of sites]}
                print(cluster_i[0])
                for key, cluster in cluster_i[1].items():
                    print(len(cluster), key, cluster)


    def neighbors_analysis_clustering_custom_made(self, verbose=False, tol_rec_distance=None):

        if tol_rec_distance is None:
            tol_rec_distance = self.symprec_group_analyzer

        sga = SpacegroupAnalyzer(self)
        system_symmetrized = sga.get_symmetrized_structure()
        self.equiv_site_idx = system_symmetrized.equivalent_indices
        self.N_unique_sites_in_unitcell = len(self.equiv_site_idx)  

        self.create_labels_for_unique_sites()

        # ---- create clusters based on their label and orbit ----

        # for each atom get its neighbors
        #   divide the neighbors depending on site_label
        #     for each of these subgroups:
        #           create an array of all their indices
        #               while there are any indices left:
        #                     take the first index
        #                       make its orbit
        #                           remove all the indices of the orbit from the array
        #                           add the orbit (--list of indexes) to the list of all orbits of atom[i]
        #           add the list of all orbits of atom[i] to the list of all clusters of atom[i]

        A = self.lattice.matrix
        inv_A = np.linalg.inv(A)

        # 3D array of indices: for each atom a list of clusters, where each cluster is a list of indices 
        all_clusters = []
        all_clusters_dist = []
        all_clusters_idx_by_dist_chem_grouped = []

        # for all  atoms in the magnetic unit cell
        for i in range(len(self)):
            # list of clusters for atom i
            clusters_i = []
            cluster_distances_i = []
            idx_cluster_by_dist_for_each_chem_i = []

            # get neighbors for the first site of the equivalent sites
            neighbors_i = self.all_neighbors_for_all_sites[i]
            rij_i_cart = self.all_neighbors_rij[i]
            # !!! worth considering to move to init and save rij_i_rec as an attribute
            rij_i_rec = np.array(rij_i_cart @ inv_A)

            # subgroups based on site_label
            for l_chem in range(self.N_unique_sites_in_unitcell):
                l_chem_label = self.labels_of_each_equiv_group[l_chem]
                # all indices of neighbors of atom i with the same label                
                chemical_subgroup = list(np.where(self.all_neighbors_labels[i,:] == l_chem_label)[0])

                clusters_i_l_chem = []
                cluster_dist_i_l_chem = []
                    # while chemical_subgroup is not empty
                while len(chemical_subgroup) > 0:
                    # j is an index for neighbors of atom i
                    # take the first index
                    j = chemical_subgroup[0]
                    # make the orbit of point j
                    point_rec = rij_i_rec[j]
                    orbit_of_point_rec = self.pg.get_orbit(point_rec)
                    # find the indices of all orbit points + index of j
                    if verbose:
                        print('lchem_label', l_chem_label)
                        print('chemical_subgroup', chemical_subgroup)
                        print('point_rec', point_rec)
                        print('orbit_of_point_rec', orbit_of_point_rec)
                    orbit_idx = []
                    for orbit_member in orbit_of_point_rec:
                        idx_orbit_member_among_neighbors = np.where(np.sum(np.abs(rij_i_rec - orbit_member), axis=1) < tol_rec_distance)[0]
                        if idx_orbit_member_among_neighbors.size > 0:
                            orbit_idx.append(idx_orbit_member_among_neighbors[0])
                        if verbose:
                            print('orbit_member', orbit_member)
                            print('orbit_idx', orbit_idx)
                    # remove all these indices from chemical_subgroup
                    for k in orbit_idx:
                        try:
                            chemical_subgroup.remove(k)
                        except ValueError:
                            warnings.warn(f"Index {k} from the orbit_idx list {orbit_idx} not able to be removed from the chemical_subgroup list {chemical_subgroup}.\nIndex listed multiple times in {orbit_idx}??")
                    # add the orbit to the list of all orbits of atom[i]
                    clusters_i_l_chem.append(orbit_idx)

                # order clusters (and related quantities) by distance from the central atom (for each chemical type individually)
                distances_i_l_chem = [np.mean(self.all_neighbors_distances[i,j_array]) for j_array in clusters_i_l_chem]    
                sort_order_i_l_chem = np.argsort(distances_i_l_chem)
                clusters_i_l_chem = [clusters_i_l_chem[i] for i in sort_order_i_l_chem]
                distances_i_l_chem = [distances_i_l_chem[i] for i in sort_order_i_l_chem]

                # append the ordered chemical clusters to
                clusters_i += clusters_i_l_chem
                cluster_distances_i += distances_i_l_chem
                # ascending array of {1, 2, ..., N_clusters_i_l_chem} for each chemical type individually
                idx_cluster_by_dist_for_each_chem_i += [i+1 for i in range(len(clusters_i_l_chem))]

            all_clusters.append(clusters_i)
            all_clusters_dist.append(cluster_distances_i)
            all_clusters_idx_by_dist_chem_grouped.append(idx_cluster_by_dist_for_each_chem_i)
        
        # list of lists of lists: for each atom, list of clusters, each cluster is a list of indices
        #   - !! should stay as lists, not numpy arrays: dimensions can differ in clusters
        self.all_clusters = all_clusters
        self.all_clusters_dist = all_clusters_dist # mean distance of cluster members from the central atom
        self.all_clusters_idx_by_dist_chem_grouped = all_clusters_idx_by_dist_chem_grouped # index of the cluster by distance for each chemical type individually

        # which cluster does each neighbor belong to?
        self.get_cluster_index_for_all()


    def create_labels_for_unique_sites(self, verbose=False):
        """CREATE LABELS FOR THE UNIQUE SITES
          1. determine if there are multiple unique site groups for each chemical element
          2. create labels for the unique sites - no index if only one group for given chemical element, added index if multiple
        
        Saves:
            self.labels_of_each_equiv_group (list): Labels of each group of equivalent sites.
        """
        element_of_each_equiv_group = [leave_only_first_part_with_letters(self[equiv_sites[0]].species_string) for equiv_sites in self.equiv_site_idx] # e.g. ['Cr', 'Cr', 'I'] if there are two non-equivalent Cr
        self.labels_of_each_equiv_group = []
        if verbose: print('element of each equiv group', element_of_each_equiv_group)
        element_occurences = {str(element): element_of_each_equiv_group.count(element) for element in np.unique(element_of_each_equiv_group)} # e.g. {'Cr': 2, 'I': 1} for the above
        if verbose: print('element occurences', element_occurences)
        element_already_occured = {str(element): 0 for element in np.unique(element_of_each_equiv_group)} # initialize to e.g. {'Cr': 0, 'I': 0}
        if verbose: print('element already occured', element_already_occured)
        for equiv_idx, element in zip(self.equiv_site_idx, element_of_each_equiv_group):
            element_already_occured[element] += 1
            species_label = element if element_occurences[element] == 1 else element + str(element_already_occured[element])
            self.labels_of_each_equiv_group.append(species_label)
            # label all the equivalent sites with the same label (e.g. Cr1 or Cr2 if more non-equivalent Cr, or simply Cr if only one equivalent Cr group)
            for i in equiv_idx:
                # change the species to a composition with species 'species_label' and occupancy 1.0
                self[i].label = species_label

    def get_total_energy_prefactors_for_pandas_table(self, verbose=False):
        # loop over all sites

        N_two_site_parameters = len(self.HH.two_site_parameters)
        N_pairs = len(self.two_site_interaction_table)

        parameter_prefactors = np.zeros((N_pairs, N_two_site_parameters), dtype=np.float64)

        for i,row in self.two_site_interaction_table.iterrows():
            if verbose:    
                print('row', row)
                print("row['center_atom_index']", row['center_atom_index'])
                print("type(row['center_atom_index'])", type(row['center_atom_index']))
            # get the two-site parameters for all pairs
            parameter_prefactors[i,:] = self.HH.two_site_energy(self.magnetic_moments[row['center_atom_index']], self.magnetic_moments[row['neighbor_index']])

        for i, parameter in enumerate(self.HH.two_site_parameters):
            self.two_site_interaction_table[parameter] = parameter_prefactors[:,i]

    def aggregate_pandas_table(self, verbose=False):
        # group by 'center_atom_index', secondarily by 'neighbor_index' and tertiary by 'cluster_index'
        # then sum all J and D
        table_grouped = self.two_site_interaction_table.groupby(['center_atom_label', 'neighbor_label', 'cluster_index'])
        
        # SUM only J and D
        table_grouped = table_grouped.agg({'distance': 'first', 'J': 'sum', 'D': 'sum'})
        self.table_grouped = table_grouped

        if verbose:
            print('TABLE SUMMED')
            print(table_grouped)
