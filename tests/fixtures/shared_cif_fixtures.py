"""Shared CIF fixture strings for local and API test suites."""

TEST_CIF_SIO2 = """# generated using pymatgen
data_SiO2
_symmetry_space_group_name_H-M   Pbcm
_cell_length_a   4.69102593
_cell_length_b   9.04290627
_cell_length_c   4.70625094
_cell_angle_alpha   90.00000000
_cell_angle_beta   90.00000000
_cell_angle_gamma   90.00000000
_symmetry_Int_Tables_number   57
_chemical_formula_structural   SiO2
_chemical_formula_sum   'Si4 O8'
_cell_volume   199.64155449
_cell_formula_units_Z   4
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
  2  '-x, -y, -z'
  3  '-x, -y, z+1/2'
  4  'x, y, -z+1/2'
  5  'x, -y+1/2, -z'
  6  '-x, y+1/2, z'
  7  '-x, y+1/2, -z+1/2'
  8  'x, -y+1/2, z+1/2'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Si  Si0  4  0.25683649  0.75000000  0.00000000  1
  O  O1  4  0.07596292  0.65774525  0.25000000  1
  O  O2  4  0.43761182  0.65761148  0.75000000  1"""

AUGMENTED_CIF_SIO2 = """<bos>
data_[Si1O2]
loop_
 _atom_type_symbol
 _atom_type_electronegativity
 _atom_type_radius
 _atom_type_ionic_radius
[
  Si  1.9000  1.1000  0.5400
  O   3.4400  0.6000  1.2600
]
_symmetry_space_group_name_H-M [P1]
_cell_length_a [5.0000]
_cell_length_b [5.0000]
_cell_length_c [5.0000]
_cell_angle_alpha [90.0000]
_cell_angle_beta [90.0000]
_cell_angle_gamma [90.0000]
_symmetry_Int_Tables_number [1]
_chemical_formula_structural [SiO2]
_chemical_formula_sum '[Si1 O2]'
_cell_volume [125.0000]
_cell_formula_units_Z [1]
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
[
  Si  Si0  1  0.0000  0.0000  0.0000  1
  O   O1   1  0.3000  0.3000  0.3000  1
  O   O2   1  0.7000  0.7000  0.7000  1
]
<eos>"""

PARTIAL_OCC_VALID_CIF = """data_Hf1Mo1Ta1W1
_symmetry_space_group_name_H-M Im-3m
_cell_length_a 4.8060
_cell_length_b 4.8060
_cell_length_c 4.8060
_cell_angle_alpha 90.0000
_cell_angle_beta 90.0000
_cell_angle_gamma 90.0000
_symmetry_Int_Tables_number 229
_chemical_formula_structural HfTaMoW
_chemical_formula_sum 'Hf1 Ta1 Mo1 W1'
_cell_volume 111.0286
_cell_formula_units_Z 1
loop_
_symmetry_equiv_pos_site_id
_symmetry_equiv_pos_as_xyz
1 '-x, z, -y'
2 '-z, x, y'
3 '-x, -z, y'
4 'z, -x, y'
5 'x, z, y'
6 '-y, -z, -x'
7 '-y, z, x'
8 '-z, -x, y'
9 'x, -z, y'
10 'z, -x, -y'
11 'y, -x, z'
12 'x, -y, -z'
13 '-y, x, z'
14 '-x, -y, z'
15 'x, y, z'
16 'y, x, -z'
17 '-x, y, -z'
18 'x, -y, z'
19 'y, x, z'
20 '-z, -x, -y'
21 '-y, x, -z'
22 '-x, y, z'
23 '-y, -x, z'
24 'y+1/2, x+1/2, -z+1/2'
25 '-x+1/2, y+1/2, -z+1/2'
26 '-y+1/2, -x+1/2, -z+1/2'
27 'z+1/2, x+1/2, y+1/2'
28 '-x+1/2, z+1/2, y+1/2'
29 '-z+1/2, -x+1/2, y+1/2'
30 'x+1/2, -z+1/2, y+1/2'
31 'z+1/2, -x+1/2, -y+1/2'
32 'x+1/2, z+1/2, -y+1/2'
33 '-z+1/2, x+1/2, -y+1/2'
34 '-x+1/2, -z+1/2, -y+1/2'
35 'y+1/2, z+1/2, x+1/2'
36 'y+1/2, -z+1/2, -x+1/2'
37 'z+1/2, y+1/2, -x+1/2'
38 '-y+1/2, z+1/2, -x+1/2'
39 '-z+1/2, -y+1/2, -x+1/2'
40 '-y+1/2, -z+1/2, x+1/2'
41 'z+1/2, -y+1/2, x+1/2'
42 '-z+1/2, y+1/2, x+1/2'
43 'y+1/2, -x+1/2, -z+1/2'
44 'x+1/2, y+1/2, -z+1/2'
45 '-y+1/2, x+1/2, -z+1/2'
46 '-x+1/2, y+1/2, z+1/2'
47 '-x+1/2, -y+1/2, z+1/2'
48 'y+1/2, -x+1/2, z+1/2'
49 'x+1/2, -y+1/2, -z+1/2'
50 'z, -y, -x'
51 'x+1/2, y+1/2, z+1/2'
52 '-y+1/2, x+1/2, z+1/2'
53 '-z, -y, x'
54 'y, -z, x'
55 'z, y, x'
56 'y, z, -x'
57 '-x+1/2, -y+1/2, -z+1/2'
58 '-y+1/2, -x+1/2, z+1/2'
59 'x+1/2, -y+1/2, z+1/2'
60 'y+1/2, x+1/2, z+1/2'
61 '-z+1/2, -x+1/2, -y+1/2'
62 'x+1/2, -z+1/2, -y+1/2'
63 'z+1/2, x+1/2, -y+1/2'
64 '-x+1/2, z+1/2, -y+1/2'
65 '-z+1/2, x+1/2, y+1/2'
66 '-x+1/2, -z+1/2, y+1/2'
67 '-z, y, -x'
68 '-y, -x, -z'
69 'z, x, y'
70 '-x, z, y'
71 'x, z, -y'
72 '-z, x, -y'
73 '-x, -z, -y'
74 'y, z, x'
75 'y, -z, -x'
76 'z, y, -x'
77 'z+1/2, -x+1/2, y+1/2'
78 'x+1/2, z+1/2, y+1/2'
79 '-y+1/2, -z+1/2, -x+1/2'
80 '-y, z, -x'
81 'x, -z, -y'
82 'z, x, -y'
83 '-z, -y, -x'
84 '-y, -z, x'
85 'z, -y, x'
86 '-z, y, x'
87 '-x, -y, -z'
88 'y, -x, -z'
89 'x, y, -z'
90 '-y+1/2, z+1/2, x+1/2'
91 '-z+1/2, -y+1/2, x+1/2'
92 'y+1/2, -z+1/2, x+1/2'
93 'z+1/2, y+1/2, x+1/2'
94 'y+1/2, z+1/2, -x+1/2'
95 '-z+1/2, y+1/2, -x+1/2'
96 'z+1/2, -y+1/2, -x+1/2'
loop_
_atom_site_type_symbol
_atom_site_label
_atom_site_symmetry_multiplicity
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Hf Hf0 1 0.0000 0.0000 0.0000 0.5
Ta Ta1 1 0.5000 0.5000 0.5000 0.5
Mo Mo2 1 0.5000 0.5000 0.5000 0.5
W W3 1 0.0000 0.0000 0.0000 0.5"""

PARTIAL_OCC_INVALID_CIF = """data_Hf1Mo1Ta1W1
_symmetry_space_group_name_H-M Im-3m
_cell_length_a 4.8099
_cell_length_b 4.8099
_cell_length_c 4.8099
_cell_angle_alpha 90.0000
_cell_angle_beta 90.0000
_cell_angle_gamma 90.0000
_symmetry_Int_Tables_number 229
_chemical_formula_structural HfTaMoW
_chemical_formula_sum 'Hf1 Ta1 Mo1 W1'
_cell_volume 111.2137
_cell_formula_units_Z 1
loop_
_symmetry_equiv_pos_site_id
_symmetry_equiv_pos_as_xyz
1 '-x, z, -y'
2 '-z, x, y'
3 '-x, -z, y'
4 'z, -x, y'
5 'x, z, y'
6 '-y, -z, -x'
7 '-y, z, x'
8 '-z, -x, y'
9 'x, -z, y'
10 'z, -x, -y'
11 'y, -x, z'
12 'x, -y, -z'
13 '-y, x, z'
14 '-x, -y, z'
15 'x, y, z'
16 'y, x, -z'
17 '-x, y, -z'
18 'x, -y, z'
19 'y, x, z'
20 '-z, -x, -y'
21 '-y, x, -z'
22 '-x, y, z'
23 '-y, -x, z'
24 'y+1/2, x+1/2, -z+1/2'
25 '-x+1/2, y+1/2, -z+1/2'
26 '-y+1/2, -x+1/2, -z+1/2'
27 'z+1/2, x+1/2, y+1/2'
28 '-x+1/2, z+1/2, y+1/2'
29 '-z+1/2, -x+1/2, y+1/2'
30 'x+1/2, -z+1/2, y+1/2'
31 'z+1/2, -x+1/2, -y+1/2'
32 'x+1/2, z+1/2, -y+1/2'
33 '-z+1/2, x+1/2, -y+1/2'
34 '-x+1/2, -z+1/2, -y+1/2'
35 'y+1/2, z+1/2, x+1/2'
36 'y+1/2, -z+1/2, -x+1/2'
37 'z+1/2, y+1/2, -x+1/2'
38 '-y+1/2, z+1/2, -x+1/2'
39 '-z+1/2, -y+1/2, -x+1/2'
40 '-y+1/2, -z+1/2, x+1/2'
41 'z+1/2, -y+1/2, x+1/2'
42 '-z+1/2, y+1/2, x+1/2'
43 'y+1/2, -x+1/2, -z+1/2'
44 'x+1/2, y+1/2, -z+1/2'
45 '-y+1/2, x+1/2, -z+1/2'
46 '-x+1/2, y+1/2, z+1/2'
47 '-x+1/2, -y+1/2, z+1/2'
48 'y+1/2, -x+1/2, z+1/2'
49 'x+1/2, -y+1/2, -z+1/2'
50 'z, -y, -x'
51 'x+1/2, y+1/2, z+1/2'
52 '-y+1/2, x+1/2, z+1/2'
53 '-z, -y, x'
54 'y, -z, x'
55 'z, y, x'
56 'y, z, -x'
57 '-x+1/2, -y+1/2, -z+1/2'
58 '-y+1/2, -x+1/2, z+1/2'
59 'x+1/2, -y+1/2, z+1/2'
60 'y+1/2, x+1/2, z+1/2'
61 '-z+1/2, -x+1/2, -y+1/2'
62 'x+1/2, -z+1/2, -y+1/2'
63 'z+1/2, x+1/2, -y+1/2'
64 '-x+1/2, z+1/2, -y+1/2'
65 '-z+1/2, x+1/2, y+1/2'
66 '-x+1/2, -z+1/2, y+1/2'
67 '-z, y, -x'
68 '-y, -x, -z'
69 'z, x, y'
70 '-x, z, y'
71 'x, z, -y'
72 '-z, x, -y'
73 '-x, -z, -y'
74 'y, z, x'
75 'y, -z, -x'
76 'z, y, -x'
77 'z+1/2, -x+1/2, y+1/2'
78 'x+1/2, z+1/2, y+1/2'
79 '-y+1/2, -z+1/2, -x+1/2'
80 '-y, z, -x'
81 'x, -z, -y'
82 'z, x, -y'
83 '-z, -y, -x'
84 '-y, -z, x'
85 'z, -y, x'
86 '-z, y, x'
87 '-x, -y, -z'
88 'y, -x, -z'
89 'x, y, -z'
90 '-y+1/2, z+1/2, x+1/2'
91 '-z+1/2, -y+1/2, x+1/2'
92 'y+1/2, -z+1/2, x+1/2'
93 'z+1/2, y+1/2, x+1/2'
94 'y+1/2, z+1/2, -x+1/2'
95 '-z+1/2, y+1/2, -x+1/2'
96 'z+1/2, -y+1/2, -x+1/2'
loop_
_atom_site_type_symbol
_atom_site_label
_atom_site_symmetry_multiplicity
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Hf Hf0 1 0.5000 0.5000 0.5000 1.0
Ta Ta1 1 0.0000 0.0000 0.0000 0.0
W W2 1 0.5000 0.5000 0.2500 1.0"""
