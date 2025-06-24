import MDAnalysis as mda
import numpy as np
from scipy.spatial.distance import cdist

def compact_membrane_iterative(input_pdb="membrane.pdb", output_pdb="membrane_compacted.pdb", target_distance=2.6):
    """
    Iteratively compact membrane by moving leaflets closer until maximum distance is 2.6 Å.
    
    - Upper leaflet moves down by 0.3 Å per iteration
    - Lower leaflet moves up by 0.3 Å per iteration
    - Total approach distance per iteration: 0.6 Å
    - Optimized for speed using vectorized operations and distance sampling
    - Updated for box dimensions: 177.4823934930 × 177.4823934930 × 150.0000000000 Å
    """
    
    try:
        # Load the universe
        u = mda.Universe(input_pdb)
        print(f"Loaded {input_pdb} successfully")
        
        # Get residue information
        total_residues = len(u.residues)
        total_atoms = len(u.atoms)
        print(f"Total residues: {total_residues}, Total atoms: {total_atoms}")
        
        # Split into leaflets
        midpoint = total_residues // 2
        all_residues = list(u.residues)
        upper_residues = all_residues[:midpoint]
        lower_residues = all_residues[midpoint:]
        
        # Get atom indices for each leaflet (for vectorized operations)
        upper_indices = []
        lower_indices = []
        
        for res in upper_residues:
            upper_indices.extend([atom.index for atom in res.atoms])
        for res in lower_residues:
            lower_indices.extend([atom.index for atom in res.atoms])
            
        upper_indices = np.array(upper_indices)
        lower_indices = np.array(lower_indices)
        
        print(f"Upper leaflet: {len(upper_indices)} atoms")
        print(f"Lower leaflet: {len(lower_indices)} atoms")
        
        # Assign chain IDs (do this once at the start)
        u.atoms[upper_indices].chainIDs = 'A'
        u.atoms[lower_indices].chainIDs = 'B'
        
        # Get initial positions
        positions = u.atoms.positions.copy()
        
        # Calculate initial maximum distance
        def calculate_max_distance_fast(pos_upper, pos_lower, sample_size=1000):
            """Fast approximation of max distance using sampling for large systems"""
            n_upper = len(pos_upper)
            n_lower = len(pos_lower)
            
            if n_upper * n_lower < 100000:  # Small system, calculate exactly
                distances = cdist(pos_upper, pos_lower)
                return np.min(distances)
            else:  # Large system, use sampling
                # Sample atoms from each leaflet
                upper_sample = np.random.choice(n_upper, min(sample_size, n_upper), replace=False)
                lower_sample = np.random.choice(n_lower, min(sample_size, n_lower), replace=False)
                
                distances = cdist(pos_upper[upper_sample], pos_lower[lower_sample])
                return np.min(distances)
        
        # Initial distance calculation
        upper_pos = positions[upper_indices]
        lower_pos = positions[lower_indices]
        current_min_distance = calculate_max_distance_fast(upper_pos, lower_pos)
        
        print(f"Initial minimum distance between leaflets: {current_min_distance:.2f} Å")
        print(f"Target distance: {target_distance:.2f} Å")
        
        iteration = 0
        max_iterations = 1000  # Safety limit
        
        # Iterative compaction
        increment = 0.3  # Movement increment in Angstroms
        while current_min_distance > target_distance and iteration < max_iterations:
            iteration += 1
            
            # Move upper leaflet down by 0.3 Å, lower leaflet up by 0.3 Å
            positions[upper_indices, 2] -= increment  # Z coordinate
            positions[lower_indices, 2] += increment
            
            # Update positions in universe
            u.atoms.positions = positions
            
            # Recalculate minimum distance
            upper_pos = positions[upper_indices]
            lower_pos = positions[lower_indices]
            current_min_distance = calculate_max_distance_fast(upper_pos, lower_pos)
            
            if iteration % 20 == 0 or current_min_distance <= target_distance:
                print(f"Iteration {iteration}: Min distance = {current_min_distance:.2f} Å")
        
        if iteration >= max_iterations:
            print(f"WARNING: Reached maximum iterations ({max_iterations})")
        else:
            print(f"SUCCESS: Target distance reached after {iteration} iterations")
        
        print(f"Final minimum distance: {current_min_distance:.2f} Å")
        print(f"Total movement: Upper leaflet down {iteration * increment:.1f} Å, Lower leaflet up {iteration * increment:.1f} Å")
        print(f"Final translation: All atoms moved up 75 Å for box centering")
        
        # Translate all atoms up by 75 Å to center in box
        print("Translating all atoms up by 75 Å to center in box...")
        positions[:, 2] += 75.0  # Move all Z coordinates up by 75 Å
        u.atoms.positions = positions
        
        # Set updated box dimensions to 10 decimal places
        # Box dimensions for APL=60 and 525 lipids per leaflet: 177.4823934930 × 177.4823934930 × 150.0000000000 Å
        box_xy = 177.4823934930  # Calculated from sqrt(525 * 60)
        box_z = 150.0000000000   # Standard Z dimension for membrane simulations
        box_dimensions = np.array([box_xy, box_xy, box_z, 90.0000000000, 90.0000000000, 90.0000000000])
        u.dimensions = box_dimensions
        
        print("=" * 80)
        print("BOX DIMENSIONS (10 DECIMAL PLACES):")
        print("=" * 80)
        print(f"X dimension: {box_dimensions[0]:.10f} Å")
        print(f"Y dimension: {box_dimensions[1]:.10f} Å")
        print(f"Z dimension: {box_dimensions[2]:.10f} Å")
        print(f"Alpha angle: {box_dimensions[3]:.10f}°")
        print(f"Beta angle:  {box_dimensions[4]:.10f}°")
        print(f"Gamma angle: {box_dimensions[5]:.10f}°")
        print("=" * 80)
        print(f"Box volume: {box_xy * box_xy * box_z:.10f} Ų")
        print(f"XY area:    {box_xy * box_xy:.10f} Ų")
        print("=" * 80)
        
        # Write the output file
        print(f"Writing to {output_pdb}...")
        with mda.Writer(output_pdb, n_atoms=total_atoms) as writer:
            writer.write(u.atoms)
        
        print(f"SUCCESS: Compacted structure saved to {output_pdb}")
        
        # Verification
        try:
            test_u = mda.Universe(output_pdb)
            chain_a_count = len(test_u.select_atoms("chainID A"))
            chain_b_count = len(test_u.select_atoms("chainID B"))
            print(f"Verification - Chain A: {chain_a_count} atoms, Chain B: {chain_b_count} atoms")
            
            # Verify box dimensions in output file
            print("Verification - Box dimensions in output file:")
            for i, label in enumerate(['X', 'Y', 'Z', 'Alpha', 'Beta', 'Gamma']):
                print(f"  {label}: {test_u.dimensions[i]:.10f}")
                
        except Exception as verify_error:
            print(f"Note: Could not verify output file: {verify_error}")
        
        return u, iteration, current_min_distance
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, 0, 0

def compact_membrane_precise(input_pdb="membrane.pdb", output_pdb="membrane_compacted.pdb", target_distance=2.6):
    """
    Alternative version that calculates exact minimum distance (slower but more accurate)
    Use this if the sampling approach gives inconsistent results
    Updated with correct box dimensions to 10 decimal places
    """
    try:
        u = mda.Universe(input_pdb)
        print(f"Loaded {input_pdb} - PRECISE MODE")
        
        total_residues = len(u.residues)
        midpoint = total_residues // 2
        all_residues = list(u.residues)
        
        upper_indices = []
        lower_indices = []
        
        for res in all_residues[:midpoint]:
            upper_indices.extend([atom.index for atom in res.atoms])
        for res in all_residues[midpoint:]:
            lower_indices.extend([atom.index for atom in res.atoms])
            
        upper_indices = np.array(upper_indices)
        lower_indices = np.array(lower_indices)
        
        # Assign chains
        u.atoms[upper_indices].chainIDs = 'A'
        u.atoms[lower_indices].chainIDs = 'B'
        
        positions = u.atoms.positions.copy()
        iteration = 0
        increment = 0.3  # Movement increment in Angstroms
        
        while iteration < 1000:
            iteration += 1
            
            # Move leaflets by 0.3 Å increments
            positions[upper_indices, 2] -= increment
            positions[lower_indices, 2] += increment
            u.atoms.positions = positions
            
            # Calculate exact minimum distance
            upper_pos = positions[upper_indices]
            lower_pos = positions[lower_indices]
            distances = cdist(upper_pos, lower_pos)
            min_distance = np.min(distances)
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Min distance = {min_distance:.2f} Å")
            
            if min_distance <= target_distance:
                break
        
        # Set box dimensions to 10 decimal places and translate
        box_xy = 177.4823934930  # sqrt(525 * 60)
        box_z = 150.0000000000
        box_dimensions = np.array([box_xy, box_xy, box_z, 90.0000000000, 90.0000000000, 90.0000000000])
        u.dimensions = box_dimensions
        
        positions[:, 2] += 75.0  # Center in box
        u.atoms.positions = positions
        
        print(f"PRECISE: Final minimum distance: {min_distance:.2f} Å after {iteration} iterations")
        print(f"PRECISE: Total movement per leaflet: {iteration * increment:.1f} Å")
        print("Box dimensions set and atoms translated up 75 Å")
        
        print("PRECISE MODE - BOX DIMENSIONS (10 DECIMAL PLACES):")
        print(f"X: {box_dimensions[0]:.10f} Å")
        print(f"Y: {box_dimensions[1]:.10f} Å") 
        print(f"Z: {box_dimensions[2]:.10f} Å")
        print(f"Angles: {box_dimensions[3]:.10f}°, {box_dimensions[4]:.10f}°, {box_dimensions[5]:.10f}°")
        
        with mda.Writer(output_pdb, n_atoms=len(u.atoms)) as writer:
            writer.write(u.atoms)
        
        return u, iteration, min_distance
        
    except Exception as e:
        print(f"ERROR in precise mode: {e}")
        return None, 0, 0

# Main execution
if __name__ == "__main__":
    input_file = "membrane.pdb"
    output_file = "membrane_compacted.pdb"
    target_dist = 2.6  # Angstroms - updated target distance
    
    print("=" * 80)
    print("ITERATIVE MEMBRANE COMPACTION")
    print("Updated for APL=60, 525 lipids per leaflet")
    print("=" * 80)
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Target distance: {target_dist} Å")
    print("-" * 80)
    
    # Use fast sampling method first
    universe, iterations, final_distance = compact_membrane_iterative(
        input_file, output_file, target_dist
    )
    
    if universe is not None:
        print("-" * 80)
        print("COMPACTION COMPLETE!")
        print(f"Iterations: {iterations}")
        print(f"Final distance: {final_distance:.2f} Å")
        
        # Optional: uncomment to use precise method for verification
        # print("\nRunning precise verification...")
        # universe2, iter2, dist2 = compact_membrane_precise(input_file, "membrane_precise.pdb", target_dist)
    else:
        print("Process failed!")
