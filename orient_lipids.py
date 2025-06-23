import numpy as np
import sys

def read_pdb(filename):
    """Read PDB file and extract ATOM records"""
    atoms = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom = {
                    'line': line,
                    'atom_name': line[12:16].strip(),
                    'residue': line[17:20].strip(),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54])
                }
                atoms.append(atom)
    return atoms

def write_pdb(atoms, filename, header_lines=None):
    """Write oriented atoms to new PDB file"""
    with open(filename, 'w') as f:
        # Write header if provided
        if header_lines:
            for line in header_lines:
                f.write(line)
        
        # Write oriented atoms
        for atom in atoms:
            # Update coordinates in the line
            line = atom['line']
            new_line = (line[:30] + 
                       f"{atom['x']:8.3f}" + 
                       f"{atom['y']:8.3f}" + 
                       f"{atom['z']:8.3f}" + 
                       line[54:])
            f.write(new_line)
        
        f.write("TER\nENDMDL\n")

def get_header_lines(filename):
    """Extract header lines from original PDB"""
    header_lines = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith(('REMARK', 'TITLE', 'CRYST1', 'MODEL')):
                header_lines.append(line)
            elif line.startswith('ATOM'):
                break
    return header_lines

def identify_functional_groups(atoms, lipid_type):
    """Identify headgroup, glycerol, and tail atoms for each lipid type"""
    headgroup_atoms = []
    glycerol_atoms = []
    tail_atoms = []
    
    if lipid_type == "S5PG":  # Phosphoglycerol
        for atom in atoms:
            name = atom['atom_name']
            # Headgroup: phosphate and attached groups
            if name in ['P1', 'O9', 'O10', 'O3', 'C5', 'C4', 'C3', 'O2', 'O1']:
                headgroup_atoms.append(atom)
            # Glycerol backbone
            elif name in ['C8', 'C7', 'C6', 'O5', 'O4', 'O6']:
                glycerol_atoms.append(atom)
            # Tails: long carbon chains
            elif name.startswith('C1') or name.startswith('C2'):
                tail_atoms.append(atom)
            else:
                glycerol_atoms.append(atom)  # Default to glycerol region
                
    elif lipid_type == "S5DG":  # Glycolipid
        for atom in atoms:
            name = atom['atom_name']
            # Headgroup: sugar moiety
            if name in ['C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 
                       'O4', 'O5', 'O6', 'O7', 'O8', 'O12']:
                headgroup_atoms.append(atom)
            # Glycerol backbone
            elif name in ['C3', 'C4', 'C5', 'C6', 'C21', 'C23', 'O1', 'O2', 'O3', 'O13', 'O14', 'O15']:
                glycerol_atoms.append(atom)
            # Tails: long carbon chains
            elif name.startswith('C1') or name.startswith('C2'):
                tail_atoms.append(atom)
            else:
                glycerol_atoms.append(atom)
                
    elif lipid_type == "S5CL":  # Cardiolipin
        for atom in atoms:
            name = atom['atom_name']
            # Headgroup: dual phosphates
            if name in ['P1', 'P2', 'O1', 'O2', 'O8', 'O9', 'O10', 'O11', 'O16', 'O17']:
                headgroup_atoms.append(atom)
            # Glycerol backbone: multiple glycerol units
            elif name in ['C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13',
                         'O3', 'O4', 'O5', 'O6', 'O7', 'O12', 'O13', 'O14', 'O15']:
                glycerol_atoms.append(atom)
            # Tails: four acyl chains
            elif (name.startswith('C1') or name.startswith('C2') or 
                  name.startswith('C3') or name.startswith('C4')):
                tail_atoms.append(atom)
            else:
                glycerol_atoms.append(atom)
                
    elif lipid_type == "S5LG":  # Lysyl phosphoglycerol
        for atom in atoms:
            name = atom['atom_name']
            # Headgroup: lysine and phosphate
            if name in ['P1', 'O9', 'O10', 'O3', 'N1', 'N2', 'C6', 'C17', 'C18']:
                headgroup_atoms.append(atom)
            # Glycerol backbone
            elif name in ['C16', 'C11', 'C10', 'C9', 'C8', 'C7', 'O1', 'O2', 'O4', 'O5', 'O11']:
                glycerol_atoms.append(atom)
            # Tails: acyl chains
            elif name.startswith('C1') or name.startswith('C2') or name in ['C3', 'C4', 'C5']:
                tail_atoms.append(atom)
            else:
                glycerol_atoms.append(atom)
    
    return headgroup_atoms, glycerol_atoms, tail_atoms

def calculate_centroid(atoms):
    """Calculate centroid of a group of atoms"""
    if not atoms:
        return np.array([0, 0, 0])
    
    coords = np.array([[atom['x'], atom['y'], atom['z']] for atom in atoms])
    return np.mean(coords, axis=0)

def orient_lipid_along_z(atoms, lipid_type):
    """Orient lipid with headgroup up, glycerol middle, tails down along Z-axis"""
    
    # Identify functional groups
    headgroup_atoms, glycerol_atoms, tail_atoms = identify_functional_groups(atoms, lipid_type)
    
    print(f"  Found {len(headgroup_atoms)} headgroup atoms")
    print(f"  Found {len(glycerol_atoms)} glycerol atoms") 
    print(f"  Found {len(tail_atoms)} tail atoms")
    
    # Calculate centroids
    head_centroid = calculate_centroid(headgroup_atoms)
    glycerol_centroid = calculate_centroid(glycerol_atoms)
    tail_centroid = calculate_centroid(tail_atoms)
    
    print(f"  Head centroid: {head_centroid}")
    print(f"  Glycerol centroid: {glycerol_centroid}")
    print(f"  Tail centroid: {tail_centroid}")
    
    # Define target vector (headgroup to tail should align with -Z direction)
    # Target: head at top (+Z), glycerol middle, tails at bottom (-Z)
    target_vector = np.array([0, 0, -1])  # Pointing down
    
    # Calculate current head-to-tail vector
    head_to_tail = tail_centroid - head_centroid
    head_to_tail_norm = head_to_tail / np.linalg.norm(head_to_tail)
    
    print(f"  Current head-to-tail vector: {head_to_tail_norm}")
    
    # Calculate rotation axis and angle
    rotation_axis = np.cross(head_to_tail_norm, target_vector)
    rotation_axis_norm = np.linalg.norm(rotation_axis)
    
    if rotation_axis_norm < 1e-6:  # Vectors are parallel
        if np.dot(head_to_tail_norm, target_vector) > 0:
            # Already aligned
            rotation_matrix = np.eye(3)
        else:
            # Need 180° rotation around any perpendicular axis
            rotation_axis = np.array([1, 0, 0])
            angle = np.pi
            rotation_matrix = rotation_matrix_from_axis_angle(rotation_axis, angle)
    else:
        rotation_axis = rotation_axis / rotation_axis_norm
        angle = np.arccos(np.clip(np.dot(head_to_tail_norm, target_vector), -1, 1))
        rotation_matrix = rotation_matrix_from_axis_angle(rotation_axis, angle)
    
    print(f"  Rotation angle: {np.degrees(angle):.1f}°")
    
    # Center molecule at glycerol backbone
    center = glycerol_centroid
    
    # Apply rotation to all atoms
    for atom in atoms:
        # Translate to origin
        coord = np.array([atom['x'], atom['y'], atom['z']]) - center
        
        # Rotate
        rotated_coord = rotation_matrix @ coord
        
        # Translate back and center at origin
        final_coord = rotated_coord
        
        # Update atom coordinates
        atom['x'] = final_coord[0]
        atom['y'] = final_coord[1] 
        atom['z'] = final_coord[2]
    
    # Final positioning: place glycerol at Z=0, headgroup positive, tails negative
    final_glycerol_centroid = calculate_centroid(glycerol_atoms)
    z_offset = -final_glycerol_centroid[2]  # Move glycerol to Z=0
    
    for atom in atoms:
        atom['z'] += z_offset
    
    # Verify final orientation
    final_head_centroid = calculate_centroid(headgroup_atoms)
    final_tail_centroid = calculate_centroid(tail_atoms)
    
    print(f"  Final head Z: {final_head_centroid[2]:.2f}")
    print(f"  Final glycerol Z: 0.00 (centered)")
    print(f"  Final tail Z: {final_tail_centroid[2]:.2f}")
    
    return atoms

def rotation_matrix_from_axis_angle(axis, angle):
    """Create rotation matrix from axis and angle using Rodrigues' formula"""
    axis = axis / np.linalg.norm(axis)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    # Rodrigues' rotation formula
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    
    R = np.eye(3) + sin_angle * K + (1 - cos_angle) * np.linalg.matrix_power(K, 2)
    return R

def main():
    """Main function to orient all lipid PDB files"""
    
    lipid_files = [
        ("02_S5PG.pdb", "S5PG"),
        ("02_S5DG.pdb", "S5DG"), 
        ("02_S5CL.pdb", "S5CL"),
        ("02_S5LG.pdb", "S5LG")
    ]
    
    print("="*80)
    print("🧬 LIPID ORIENTATION SCRIPT")
    print("="*80)
    print("Orienting lipids along Z-axis:")
    print("• Headgroups facing UP (+Z direction)")
    print("• Glycerol backbone at CENTER (Z=0)")  
    print("• Tail groups facing DOWN (-Z direction)")
    print("• Molecular axis aligned with Z-vector")
    print()
    
    for filename, lipid_type in lipid_files:
        print(f"Processing {filename} ({lipid_type})...")
        
        try:
            # Read original PDB
            atoms = read_pdb(filename)
            header_lines = get_header_lines(filename)
            
            print(f"  Read {len(atoms)} atoms")
            
            # Orient the lipid
            oriented_atoms = orient_lipid_along_z(atoms, lipid_type)
            
            # Write oriented PDB
            output_filename = f"oriented_{filename}"
            write_pdb(oriented_atoms, output_filename, header_lines)
            
            print(f"  ✅ Saved oriented structure to {output_filename}")
            print()
            
        except Exception as e:
            print(f"  ❌ Error processing {filename}: {e}")
            print()
    
    print("="*80)
    print("✅ LIPID ORIENTATION COMPLETE!")
    print("="*80)
    print()
    print("Generated files:")
    for filename, _ in lipid_files:
        print(f"• oriented_{filename}")
    print()
    print("Next step: Use these oriented files in the PACKMOL script")
    print("for proper membrane assembly!")

if __name__ == "__main__":
    main()
