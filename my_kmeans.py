
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import functools
import math
import os
import re
import sys



from itertools import combinations, compress
from scipy.spatial import distance
from scipy import signal
from sklearn.cluster import KMeans
from sklearn import metrics




X_H_COVALENT = 1.1 #X = C, O, N
X_H_HB = 2.5 #X = O,N
wdir = os.getcwd()
C_N = 1.4
C_O = 1.25
N_H = 1.05

def read_reference(conformation_fname):
  """
  Read from a ',' split key-atom file and return a list of int
  remove duplicates
  """

  #Create empty set
  reference_atoms = []

  #Try/catch if file cannot be found. Open file in read mode
  #For eveyr line in the text file, strip all white spaces from front and back
  #If not empty line, split line on commas and put integers in set. These correspond to atom numbers of the key atoms
  #Return this list

  try:
      with open(conformation_fname, "r") as fin :
          num = 1
          for line in fin:
            if num < 10:
                num = num + 1
                continue
            content = line.strip()
            if content == '':
                continue
            else:
                reference_atoms.append(content.split())
            #reference_atom_num.update([int(i) for i in content.split(',')])
          return reference_atoms
  #Catch OS error
  except OSError:
    print('OS error')
    sys.exit()
  #Catch value error (not appropriate values to be converted to int)
  except ValueError:
    print('Could not convert data to integer')
    sys.exit()


#We have 2d array, inner arrays contain atom # and x,y,z
# Goal is to calculate dihedreal angle at peptide bonds (CONH)
#How to even tell what are conh if they are out of order

def getCoords(reference_atom_list, ind_atom):
    atom_list = reference_atom_list[ind_atom][1:]
    final_list = [float(coord) for coord in atom_list]
    return final_list

def getDist(coord1, coord2):
    x_change = coord1[0] - coord2[0]
    y_change = coord1[1] - coord2[1]
    z_change = coord1[2] - coord2[2]
    dist = math.sqrt(x_change*x_change + y_change*y_change + z_change*z_change)
    return dist

#See if there is any way to clear this up
def id_peptide_bonds_O(reference_atom_list, ind_C):
            for atom2 in reference_atom_list:
                if atom2[0] == 'O':
                    dist = getDist(getCoords(reference_atom_list, ind_C), getCoords(reference_atom_list, reference_atom_list.index(atom2)))
                    if (dist < C_O):
                        return True
            return False

#See if there is any way to clear this up
def id_peptide_bonds_N(reference_atom_list):
    list_of_peptide_carbons = []
    list_of_peptide_nitrogens = []
    for atom1 in reference_atom_list:
        if atom1[0] == 'C':
            for atom2 in reference_atom_list:
                if atom2[0] == 'N':
                    dist = getDist(getCoords(reference_atom_list, reference_atom_list.index(atom1)), getCoords(reference_atom_list, reference_atom_list.index(atom2)))
                    if (dist < C_N):
                        for atom_3 in reference_atom_list:
                            if atom_3[0] =='O':
                                dist2 = getDist(getCoords(reference_atom_list, reference_atom_list.index(atom1)), getCoords(reference_atom_list, reference_atom_list.index(atom_3)))
                                if (dist2 < C_O):
                                    for atom4 in reference_atom_list:
                                        if atom4[0] == "H":
                                            dist3 = getDist(getCoords(reference_atom_list, reference_atom_list.index(atom2)), getCoords(reference_atom_list, reference_atom_list.index(atom4)))
                                            if (dist3 < N_H):
                                                list_of_peptide_carbons.append(tuple([reference_atom_list.index(atom1), reference_atom_list.index(atom2), reference_atom_list.index(atom_3), reference_atom_list.index(atom4)]))

    return list_of_peptide_carbons


def check_same_carbon(peptide1, peptide2):
    if peptide1[0]==peptide2[0]:
        return True
    else:
        return False

def process_list_of_pep_cs(list_of_peptide_carbons):
    for pep_c in list_of_peptide_carbons:
        pep_c_ind = list_of_peptide_carbons.index(pep_c)
        if pep_c_ind != len(list_of_peptide_carbons)-1:
            for pep_c2 in list_of_peptide_carbons[(pep_c_ind+1):]:
                if check_same_carbon(pep_c, pep_c2)== True:
                    list_of_peptide_carbons.remove(pep_c)
                    list_of_peptide_carbons.remove(pep_c2)
    return(list_of_peptide_carbons)



def vector_subtraction(coord1, coord2):
    c1 = np.array(coord1)
    c2 = np.array(coord2)
    b_final = c1-c2
    return b_final

def calculate_dihedral(reference_atom_list, list_of_peptide_carbons):
    list_of_dihedral = []
    for peptide_group in list_of_peptide_carbons:
        C_pos = peptide_group[0]
        C_coord = getCoords(reference_atom_list, C_pos)
        N_pos = peptide_group[1]
        N_coord = getCoords(reference_atom_list, N_pos)
        O_pos = peptide_group[2]
        O_coord = getCoords(reference_atom_list, O_pos)
        H_pos = peptide_group[3]
        H_coord = getCoords(reference_atom_list, H_pos)
        vec_b1 = vector_subtraction(C_coord, O_coord)
        vec_b2 = vector_subtraction(N_coord, C_coord)
        vec_b3 = vector_subtraction(H_coord, N_coord)
        n1 = np.cross(vec_b1, vec_b2)
        n2 = np.cross(vec_b2, vec_b3)
        m1 = np.cross(n1, vec_b2)
        x = np.dot(n1,n2)
        y = np.dot(m1,n2)
        dihedral = math.atan2(y,x) * (180/math.pi)
        list_of_dihedral.append(dihedral)
    return list_of_dihedral

if __name__ == '__main__':
    test_file = "dft_snap_1.txt"
    atoms = read_reference(test_file)
    #print(atoms)

    pep_cs = id_peptide_bonds_N(atoms)
    pep_cs_cleaned = process_list_of_pep_cs(pep_cs)
    #print(pep_cs)
    #print(pep_cs_cleaned)
    dihedrals = calculate_dihedral(atoms, pep_cs_cleaned)
    print(dihedrals)


