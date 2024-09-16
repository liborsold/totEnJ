
# !!!! conda activate aiida_ase
import os
import numpy as np
import shutil

def prepare_spin_states(states, template_folder, target_folder, magmom_generator_function):
    import os

    curdir = os.getcwd() 

    for i_state, state in enumerate(states):
        new_folder = target_folder + f'/state_{i_state:02d}/'
        # copy template folder to target folder
        shutil.copytree(template_folder, new_folder)

        MAGMOM = magmom_generator_function(state)
        # do python version of sed -i "s/MAGMOM = .*/MAGMOM = {MAGMOM}/" INCAR
        with open(new_folder + 'INCAR', 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if 'MAGMOM' in line:
                lines[i] = f'  MAGMOM = {MAGMOM}\n'
            

        with open(new_folder + 'INCAR', 'w') as f:
            f.writelines(lines)

        os.chdir(new_folder)
        os.system('qsub vasp_launch')
        os.chdir(curdir)


def magmom_CrTe(state, magnitude=3.5):
    cz, *m_vec = state
    n_Cr = len(m_vec)
    m = np.array([magnitude, 0, 0]) if cz == 0 else np.array([0, 0, magnitude])
    MAGMOM = np.array([m * m_vec[i] for i in range(n_Cr)]).flatten()
    return ' '.join([str(magmom) for magmom in MAGMOM]) + f' {2*3*n_Cr}*0'


def main():

    path_template = '/W/lv268562/2D/CrXY/CrXY_template/exchange_from_total_energy/template_4_1_1/'

    # ======= PARAMETERS =======
    materials = ['CrTe2'] #['CrS2', 'CrSe2', 'CrTe2', 'CrSSe', 'CrSTe', 'CrSeTe']
    Hubbard_Us = [3.0] #[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    path_template = '/W/lv268562/2D/CrXY/CrXY_template/exchange_from_total_energy/template_4_1_1/'

    # ======= END OF PARAMETERS =======

    states = np.loadtxt(path_template + 'spin_states.csv', delimiter=',')

    for material in materials:
        for U in Hubbard_Us:
            print(f'Preparing {material} with U = {U:.1f} eV')
            path = f'/W/lv268562/2D/CrXY/{material}/U_{U:.1f}/exchange_from_total_energy/4x1x1/'
            try:
                os.mkdir(path)
            except:
                pass
            os.chdir(path)

            shutil.copytree(path_template, path + 'template_4_1_1/')

            os.chdir('template_4_1_1/')

            # copy CONTCAR and POTCAR from ../../relax
            os.system('cp ../../../relax/CONTCAR .')
            os.system('cp ../../../relax/POTCAR .')

            # build supercell
            os.system('python ASE_build_supercell.py CONTCAR 4 1 1')

            # change Hubbard Hubbard_U in INCAR by f'{U:.1f}' with regex
            with open('INCAR', 'r') as f:
                lines = f.readlines()
            for i in range(len(lines)): 
                lines[i] = lines[i].replace('Hubbard_U', f'{U:.1f}')
            with open('INCAR', 'w') as f:
                f.writelines(lines)

            # prepare and launch all the spin-state calculations
            prepare_spin_states(states, path+'template_4_1_1/', path, magmom_CrTe)


if __name__ == '__main__':
    main()