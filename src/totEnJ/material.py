from pymatgen.core import Lattice, Structure, Molecule
from pymatgen.vis.structure_vtk import StructureVis
from os.path import exists
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from matplotlib.patches import Circle, Polygon

class StructureJ(Structure):

    def set_magnetic_atoms(self, magnetic_atoms, discard_nonmagnetic_atoms=True):
        self.magnetic_atoms = magnetic_atoms

    def set_discard_nonmagnetic_atoms(self, discard_nonmagnetic_atoms):
        self.discard_nonmagnetic_atoms = discard_nonmagnetic_atoms

    def set_magnetic_supercell(self, magnetic_supercell):
        self.magnetic_supercell = magnetic_supercell

    def set_supercell_out_name(self, supercell_out_name):
        self.supercell_out_name = supercell_out_name
    
    def remove_nonmagnetic_atoms(self):
        self.remove_sites([i for i in range(len(self)) if i not in self.magnetic_atoms])

    def make_supercell(self):
        super().make_supercell(self.magnetic_supercell)
        # label atoms by atom type + index
        for i, site in enumerate(self):
            site.species = site.species_string + str(i+1)
        self.to(self.supercell_out_name)
        
    def show_supercell_now(self):
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

    def get_J1_with_phantoms_for_supercell(self, four_state_atoms_indeces=(0,1), neighbor_cutoff=15.0, round_decimals=3):
        
        self.neighbor_cutoff = neighbor_cutoff
        self.round_decimals = round_decimals
                
        def nn_order_from_distances(distances):
            """Convert an array of distances to an array classifying the order of nearest neighbors based purely on these distances.

            Args:
                distances (array-like): List of distances to neighbors.

            Returns:
                numpy array: order of nearest neighbors (starting from 1)
            """
            distances = np.array(distances)
            # find unique distances and sort in ascending order
            unique_distances = np.sort(np.unique(distances.round(decimals=self.round_decimals)))
            # dictionary of neighbor labels by distance\
            nn_order_by_distance = {}
            for i, distance in enumerate(unique_distances):
                # 1st nearest neighbor labeled starting from 1 (not from 0)
                nn_order_by_distance[distance] = i+1
            # map distances to nn_order_by_distance
            nn_order = np.vectorize(nn_order_by_distance.get)(distances.round(decimals=self.round_decimals))
            return nn_order


        def count_nn_order_neighbors(neighbors_of_id1_nn_order):
            """Given a list of nearest neighbor orders, count the number of neighbors of each order.

            Args:
                neighbors_of_id1_nn_order (array-like): List of nearest neighbor orders.

            Returns:
                list: List of counts of neighbors of each order; e.g., [2, 4, 0, 8] means there are 2 neighbors of order 1 (i.e., 1st-nearest neighbors), 4 neighbors of order 2, 0 neighbors of order 3, and 8 neighbors of order 4.
            """
            return [int(np.sum(neighbors_of_id1_nn_order == i)) for i in range(1, int(max(neighbors_of_id1_nn_order))+1)]

        # find neighbors: neighbors is a list (for each site in the unit cell) of list of PeriodicNeighbor objects ... https://pymatgen.org/pymatgen.core.html#pymatgen.core.structure.PeriodicNeighbor
            # the return type is a [(site, distance) …]
        all_neighbors_for_all_sites = self.get_all_neighbors(self.neighbor_cutoff)

        # for the first four-state-method atom find all his second-type four-state-method-atom friends
        #   : identify how many and which order nearest-neighbor atoms they are

        id1 = four_state_atoms_indeces[0]
        id2 = four_state_atoms_indeces[1]

        id1_coords = np.array(self[id1].coords)
        all_neighbors_for_id1 = all_neighbors_for_all_sites[id1]

        neighbors_of_id1_coords = np.array( [neighbor.coords for neighbor in all_neighbors_for_id1] )
        neighbors_of_id1_labels = [neighbor.species_string for neighbor in all_neighbors_for_id1]
        neighbors_of_id1_distances = np.linalg.norm(neighbors_of_id1_coords - id1_coords, axis=1)
        neighbors_of_id1_nn_order = nn_order_from_distances(neighbors_of_id1_distances)
        neighbors_of_id1_nn_number = count_nn_order_neighbors(neighbors_of_id1_nn_order)

        neighbor_is_type_id2 = [neighbor.species_string == self[id2].species_string for neighbor in all_neighbors_for_id1]
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
        all_neighbors_for_id2 = all_neighbors_for_all_sites[id2]

        neighbors_of_id2_coords = np.array( [neighbor.coords for neighbor in all_neighbors_for_id2] )
        neighbors_of_id2_labels = [neighbor.species_string for neighbor in all_neighbors_for_id2]
        neighbors_of_id2_distances = np.linalg.norm(neighbors_of_id2_coords - id2_coords, axis=1)
        neighbors_of_id2_nn_order = nn_order_from_distances(neighbors_of_id2_distances)
        neighbors_of_id2_nn_number = count_nn_order_neighbors(neighbors_of_id2_nn_order)

        neighbor_is_type_id1 = [neighbor.species_string == self[id1].species_string for neighbor in all_neighbors_for_id2]
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
        
        # save data to object
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
        plt.ylim(0, ylim)
        plt.xlim(0, self.neighbor_cutoff*1.05)
        # y ticks in steps of 2
        plt.yticks(np.arange(ax.get_ylim()[0], ax.get_ylim()[1], 2))
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
    

