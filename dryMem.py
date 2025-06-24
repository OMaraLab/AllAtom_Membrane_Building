import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
import numpy as np

# Load the structure
u = mda.Universe('solv.pdb')

# Define selections
leaflet_A_phosphates = u.select_atoms('segid A and name P1 P2')
leaflet_B_phosphates = u.select_atoms('segid B and name P1 P2')
leaflet_A = u.select_atoms('segid A and not resname SOL')
leaflet_B = u.select_atoms('segid B and not resname SOL')
membrane = leaflet_A + leaflet_B
water = u.select_atoms('resname SOL')

print(f"Total atoms: {len(u.atoms)}")
print(f"Leaflet A P1/P2 atoms: {len(leaflet_A_phosphates)}")
print(f"Leaflet B P1/P2 atoms: {len(leaflet_B_phosphates)}")
print(f"Leaflet A atoms: {len(leaflet_A)}")
print(f"Leaflet B atoms: {len(leaflet_B)}")
print(f"Total membrane atoms: {len(membrane)}")
print(f"Water molecules: {len(water.residues)}")

# Calculate center of mass for P1/P2 atoms in each leaflet
com_A = leaflet_A_phosphates.center_of_mass()
com_B = leaflet_B_phosphates.center_of_mass()

print(f"Leaflet A P1/P2 COM: {com_A}")
print(f"Leaflet B P1/P2 COM: {com_B}")

# Determine which coordinate axis is the membrane normal (largest difference)
diff = np.abs(com_A - com_B)
membrane_normal_axis = np.argmax(diff)
axis_names = ['X', 'Y', 'Z']
print(f"Membrane normal axis: {axis_names[membrane_normal_axis]} (difference: {diff[membrane_normal_axis]:.2f} Å)")

# Determine bounds between leaflets
min_coord = min(com_A[membrane_normal_axis], com_B[membrane_normal_axis])
max_coord = max(com_A[membrane_normal_axis], com_B[membrane_normal_axis])

print(f"Water will be deleted between {min_coord:.2f} and {max_coord:.2f} Å along {axis_names[membrane_normal_axis]}-axis")

# Memory-efficient approach: process water molecules in chunks
chunk_size = 1000
water_residues = water.residues
total_water = len(water_residues)
water_to_remove = []

print(f"Processing {total_water} water molecules in chunks of {chunk_size}...")

for i in range(0, total_water, chunk_size):
    end_idx = min(i + chunk_size, total_water)
    chunk_residues = water_residues[i:end_idx]
    
    # Get center of mass for this chunk of water molecules
    water_com_chunk = np.array([res.atoms.center_of_mass() for res in chunk_residues])
    
    # Check if water molecules are between the leaflets
    water_coords = water_com_chunk[:, membrane_normal_axis]
    between_leaflets = (water_coords >= min_coord) & (water_coords <= max_coord)
    
    # Only check distances for water molecules between leaflets
    if np.any(between_leaflets):
        # Calculate distances for this chunk
        dist_array = distance_array(water_com_chunk, membrane.positions, box=u.dimensions)
        
        # Find minimum distance to membrane for each water molecule in chunk
        min_distances = np.min(dist_array, axis=1)
        
        # Water molecules to remove: within 8 Å AND between leaflets
        within_cutoff = min_distances <= 8.0
        to_remove_in_chunk = between_leaflets & within_cutoff
        
        # Add residue indices to removal list
        for j, remove in enumerate(to_remove_in_chunk):
            if remove:
                water_to_remove.append(i + j)
    
    print(f"Processed {end_idx}/{total_water} water molecules")

# Convert to boolean mask
water_to_keep_mask = np.ones(total_water, dtype=bool)
water_to_keep_mask[water_to_remove] = False

water_deleted = len(water_to_remove)
water_kept = np.sum(water_to_keep_mask)

print(f"Water molecules deleted (within 8 Å AND between leaflets): {water_deleted}")
print(f"Water molecules kept: {water_kept}")

# Get atoms of water molecules to keep
water_residues_to_keep = water_residues[water_to_keep_mask]
water_atoms_to_keep = sum([res.atoms for res in water_residues_to_keep], u.atoms[[]])

# Create final selection (membrane + remaining water)
final_selection = membrane + water_atoms_to_keep

print(f"Final system size: {len(final_selection)} atoms")

# Write the output
final_selection.write('solv_drymem.pdb')
print("Output written to solv_drymem.pdb")
