#!/usr/bin/env python3
"""
Enhanced Membrane Builder with Chain Labeling
==============================================

This script generates PACKMOL input for touching-tail lipid membranes with:
- Precise headgroup alignment across leaflets
- Proper chain labeling (A=upper, B=lower leaflet)
- Comprehensive documentation and parameter explanations
- Support for multiple lipid types with specific positioning

The membrane design features:
- Phosphate-to-phosphate distance of 40Å between leaflets
- Sugar groups aligned with phosphate planes for consistency
- Lysine groups included in S5LG headgroup positioning
- Touching tails without interdigitation
- 55Å total membrane width to accommodate lysine extension

USAGE:
    python3 genPackMol.py --pdb <files> --numbers <counts> --apl <value> --target_total <total>

EXAMPLE:
    python3 genPackMol.py --pdb oriented_02_S5PG.pdb oriented_02_S5LG.pdb \\
                         --numbers 35 17 --apl 200 --target_total 52

Author: Enhanced Membrane Builder
Date: 2024
"""

import sys
import math
import os
import argparse
import numpy as np

# ============================================================================
# COMMAND-LINE ARGUMENT PARSING
# ============================================================================

def parse_arguments():
    """
    Parse and validate command-line arguments for membrane generation.
    
    Returns:
        argparse.Namespace: Parsed arguments containing:
            - pdb: List of input PDB files
            - numbers: List of lipid counts per leaflet
            - apl: Area per lipid in Ų
            - target_total: Target total lipids per leaflet
    """
    parser = argparse.ArgumentParser(
        description="Generate PACKMOL input for touching-tail membrane assembly with chain labeling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MEMBRANE DESIGN SPECIFICATIONS:
  ✓ Headgroup Alignment: All phosphate/sugar groups positioned at same Z-planes
  ✓ Chain Labeling: Upper leaflet = Chain A, Lower leaflet = Chain B
  ✓ P1-P1 Distance: 40Å separation between phosphate groups
  ✓ Membrane Width: 55Å total width accommodates lysine extension
  ✓ Tail Contact: Touching tails without interdigitation for stability
  ✓ Multi-lipid Support: S5PG, S5LG, S5DG, S5CL with proper positioning

SUPPORTED LIPID TYPES:
  • S5PG: Phosphatidylglycerol (phosphate headgroup)
  • S5LG: Lysylphosphatidylglycerol (phosphate + lysine headgroup)
  • S5DG: Digalactosyldiacylglycerol (sugar headgroup)
  • S5CL: Cardiolipin (dual phosphate headgroups)

EXAMPLE USAGE:
  python3 genPackMol.py --pdb oriented_02_S5PG.pdb oriented_02_S5LG.pdb \\
                       --numbers 35 17 --apl 200 --target_total 52
        """)
    
    parser.add_argument('--pdb', nargs='+', required=True,
                        help='Input PDB files (space-separated list)')
    parser.add_argument('--numbers', nargs='+', type=int, required=True,
                        help='Number of each lipid type per leaflet (must match PDB count)')
    parser.add_argument('--apl', type=float, default=70.0,
                        help='Area per lipid in Ų (default: 70, typical range: 50-200)')
    parser.add_argument('--target_total', type=int, default=52,
                        help='Target total lipids per leaflet (default: 52)')
    
    return parser.parse_args()

# ============================================================================
# PDB FILE INPUT/OUTPUT OPERATIONS
# ============================================================================

def read_pdb_atoms(filename):
    """
    Read ATOM records from a PDB file and extract coordinate information.
    
    This function parses PDB format files to extract atomic coordinates
    and identifiers needed for membrane assembly.
    
    Args:
        filename (str): Path to input PDB file
        
    Returns:
        list: List of dictionaries containing atom information:
            - line: Original PDB line
            - name: Atom name (4 characters, stripped)
            - x, y, z: Atomic coordinates as floats
            
    Raises:
        SystemExit: If file not found or invalid PDB format
    """
    atoms = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    # Parse PDB ATOM record according to standard format
                    # Columns 13-16: Atom name
                    # Columns 31-38: X coordinate
                    # Columns 39-46: Y coordinate  
                    # Columns 47-54: Z coordinate
                    atom = {
                        'line': line,
                        'name': line[12:16].strip(),
                        'x': float(line[30:38]),
                        'y': float(line[38:46]),
                        'z': float(line[46:54])
                    }
                    atoms.append(atom)
    except FileNotFoundError:
        print(f"❌ ERROR: Could not find PDB file {filename}")
        print(f"   Please ensure the file exists in the current directory")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ ERROR: Invalid PDB format in {filename}")
        print(f"   Error details: {e}")
        print(f"   Please check that coordinates are properly formatted")
        sys.exit(1)
    
    if not atoms:
        print(f"⚠️  WARNING: No ATOM records found in {filename}")
    
    return atoms

def write_pdb_atoms(atoms, filename, header_lines=None, chain_id='A'):
    """
    Write atoms to PDB file with proper chain labeling and formatting.
    
    This function creates properly formatted PDB files with chain identifiers
    for membrane leaflet identification (A=upper, B=lower).
    
    Args:
        atoms (list): List of atom dictionaries with coordinates
        filename (str): Output PDB filename
        header_lines (list, optional): Header lines to include
        chain_id (str): Chain identifier ('A' for upper, 'B' for lower leaflet)
    """
    with open(filename, 'w') as f:
        # Write header information if provided
        if header_lines:
            f.write("REMARK Enhanced Membrane Builder - Chain Labeled Output\n")
            f.write(f"REMARK Chain {chain_id}: {'Upper' if chain_id == 'A' else 'Lower'} Leaflet\n")
            for line in header_lines:
                f.write(line)
        
        # Write atoms with updated coordinates and chain ID
        for i, atom in enumerate(atoms, 1):
            line = atom['line']
            
            # Update coordinates in PDB format (columns 31-54)
            # Update chain ID (column 22)
            # Update atom serial number (columns 7-11)
            new_line = (
                f"{line[:6]}{i:5d} "  # ATOM + serial number
                f"{line[12:21]}{chain_id}"  # Atom info + chain ID
                f"{line[22:30]}"  # Residue info
                f"{atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}"  # Coordinates
                f"{line[54:]}"  # Remaining fields
            )
            f.write(new_line)
        
        # Proper PDB termination
        f.write("TER\n")
        f.write("ENDMDL\n")

def get_pdb_header(filename):
    """
    Extract header lines from PDB file for preservation in output.
    
    Header lines contain important metadata like crystal information,
    remarks, and titles that should be preserved in oriented files.
    
    Args:
        filename (str): Input PDB filename
        
    Returns:
        list: List of header lines to preserve
    """
    header_lines = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                # Preserve important header records
                if line.startswith(('REMARK', 'TITLE', 'CRYST1', 'MODEL')):
                    header_lines.append(line)
                elif line.startswith('ATOM'):
                    break  # Stop at first ATOM record
    except FileNotFoundError:
        pass  # File might not exist, return empty headers
    return header_lines

# ============================================================================
# LIPID STRUCTURE ANALYSIS AND CLASSIFICATION
# ============================================================================

def identify_lipid_groups(atoms, lipid_type):
    """
    Identify and classify atoms into functional groups for membrane positioning.
    
    This function categorizes atoms based on lipid type to enable precise
    positioning of headgroups and tails during membrane assembly.
    
    HEADGROUP DEFINITIONS BY LIPID TYPE:
    • S5PG (Phosphatidylglycerol): Phosphate group only (P1, O9, O10, O3)
    • S5LG (Lysylphosphatidylglycerol): Phosphate + lysine extension (includes N1, N2, C6, C17, C18)
    • S5DG (Digalactosyldiacylglycerol): Outermost sugar ring only (C13, C14, O8, O12)  
    • S5CL (Cardiolipin): Both phosphate groups (P1, P2, and associated oxygens)
    
    RATIONALE FOR HEADGROUP SELECTION:
    - Only atoms that define the membrane interface are included
    - Lysine groups in S5LG are included because they extend into solution
    - Sugar groups use only outermost atoms to avoid internal bulk
    - This ensures proper alignment across different lipid types
    
    Args:
        atoms (list): List of atom dictionaries
        lipid_type (str): Lipid type identifier (S5PG, S5LG, S5DG, S5CL)
        
    Returns:
        tuple: (headgroup_atoms, tail_atoms, other_atoms)
    """
    headgroup_atoms = []
    tail_atoms = []
    other_atoms = []
    
    for atom in atoms:
        name = atom['name']
        
        if lipid_type == "S5PG":  
            # Phosphatidylglycerol: phosphate group defines membrane interface
            if name in ['P1', 'O9', 'O10', 'O3']:
                headgroup_atoms.append(atom)
            elif name.startswith(('C1', 'C2')):  # Acyl chains
                tail_atoms.append(atom)
            else:
                other_atoms.append(atom)
                
        elif lipid_type == "S5LG":  
            # Lysylphosphatidylglycerol: phosphate + lysine extension
            # Lysine included because it extends significantly into solution
            if name in ['P1', 'O9', 'O10', 'O3', 'N1', 'N2', 'C6', 'C17', 'C18']:
                headgroup_atoms.append(atom)
            elif name.startswith(('C1', 'C2')):  # Acyl chains
                tail_atoms.append(atom)
            else:
                other_atoms.append(atom)
                
        elif lipid_type == "S5DG":  
            # Digalactosyldiacylglycerol: outermost sugar ring only
            # Using only terminal sugar atoms to avoid internal bulk
            if name in ['C13', 'C14', 'O8', 'O12']:
                headgroup_atoms.append(atom)
            elif name.startswith(('C1', 'C2')):  # Acyl chains
                tail_atoms.append(atom)
            else:
                other_atoms.append(atom)
                
        elif lipid_type == "S5CL":  
            # Cardiolipin: both phosphate groups define interface
            if name in ['P1', 'P2', 'O1', 'O2', 'O8', 'O9', 'O10', 'O11', 'O16', 'O17']:
                headgroup_atoms.append(atom)
            elif any(name.startswith(x) for x in ['C1', 'C2', 'C3', 'C4']):  # Four acyl chains
                tail_atoms.append(atom)
            else:
                other_atoms.append(atom)
        else:
            # Unknown lipid type: use heuristic classification
            # Phosphorus or numbered oxygens likely headgroup
            if 'P' in name or ('O' in name and any(c.isdigit() for c in name)):
                headgroup_atoms.append(atom)
            elif name.startswith('C') and any(c.isdigit() for c in name):
                tail_atoms.append(atom)
            else:
                other_atoms.append(atom)
    
    return headgroup_atoms, tail_atoms, other_atoms

def analyze_lipid_structure(atoms, lipid_type):
    """
    Analyze lipid structure to determine key dimensions for membrane assembly.
    
    This analysis provides critical information for positioning lipids
    in the membrane to achieve proper headgroup alignment and tail contact.
    
    ANALYSIS COMPONENTS:
    • Center of mass: Overall molecular position reference
    • Headgroup span: Range of headgroup atoms (interface definition)
    • Tail span: Range of tail atoms (hydrophobic core)
    • Total span: Complete molecular dimensions
    
    This information is used to calculate precise positioning targets
    that ensure proper membrane architecture.
    
    Args:
        atoms (list): List of atom dictionaries
        lipid_type (str): Lipid type identifier
        
    Returns:
        dict: Analysis results containing dimensional information
    """
    if not atoms:
        return None
    
    # Calculate overall molecular dimensions
    all_z = [atom['z'] for atom in atoms]
    com_z = sum(all_z) / len(all_z)  # Center of mass Z-coordinate
    
    # Classify atoms into functional groups
    headgroup_atoms, tail_atoms, other_atoms = identify_lipid_groups(atoms, lipid_type)
    
    # Analyze headgroup positioning (membrane interface)
    if headgroup_atoms:
        head_z = [atom['z'] for atom in headgroup_atoms]
        head_min, head_max = min(head_z), max(head_z)
        head_center = (head_min + head_max) / 2.0
    else:
        head_min = head_max = head_center = com_z
        print(f"⚠️  WARNING: No headgroup atoms identified for {lipid_type}")
    
    # Analyze tail positioning (hydrophobic core)
    if tail_atoms:
        tail_z = [atom['z'] for atom in tail_atoms]
        tail_min, tail_max = min(tail_z), max(tail_z)
        tail_center = (tail_min + tail_max) / 2.0
    else:
        tail_min = tail_max = tail_center = com_z
        print(f"⚠️  WARNING: No tail atoms identified for {lipid_type}")
    
    return {
        'lipid_type': lipid_type,
        'com_z': com_z,
        'total_span': max(all_z) - min(all_z),
        'headgroup_min': head_min,
        'headgroup_max': head_max,
        'headgroup_center': head_center,
        'tail_min': tail_min,
        'tail_max': tail_max,
        'tail_center': tail_center,
        'head_to_tail_vector': tail_center - head_center
    }

# ============================================================================
# LIPID ORIENTATION FOR MEMBRANE ASSEMBLY
# ============================================================================

def calculate_centroid(atoms):
    """
    Calculate the geometric centroid of a group of atoms.
    
    Args:
        atoms (list): List of atom dictionaries with x, y, z coordinates
        
    Returns:
        numpy.ndarray: 3D centroid coordinates [x, y, z]
    """
    if not atoms:
        return np.array([0.0, 0.0, 0.0])
    
    coords = np.array([[atom['x'], atom['y'], atom['z']] for atom in atoms])
    return coords.mean(axis=0)

def orient_lipid_upper_leaflet(atoms, lipid_type):
    """
    Orient lipid for upper leaflet positioning (headgroup up, tails down).
    
    ORIENTATION STRATEGY:
    • Headgroups positioned toward positive Z (solution interface)
    • Tails positioned toward negative Z (membrane center)  
    • Maintains proper membrane polarity
    
    This ensures that when placed in the upper leaflet, headgroups
    face the upper solution and tails point toward the membrane center.
    
    Args:
        atoms (list): List of atom dictionaries
        lipid_type (str): Lipid type identifier
        
    Returns:
        list: Oriented atom list for upper leaflet
    """
    atoms_copy = [atom.copy() for atom in atoms]
    
    # Identify functional groups for orientation analysis
    headgroup_atoms, tail_atoms, other_atoms = identify_lipid_groups(atoms_copy, lipid_type)
    
    if not headgroup_atoms or not tail_atoms:
        print(f"⚠️  WARNING: Could not identify headgroup/tail atoms for {lipid_type}")
        print(f"   Using original orientation")
        return atoms_copy
    
    # Calculate group centroids
    head_centroid = calculate_centroid(headgroup_atoms)
    tail_centroid = calculate_centroid(tail_atoms)
    
    # Determine head-to-tail vector
    head_to_tail = tail_centroid - head_centroid
    
    # For upper leaflet: headgroup should be above tails (positive Z)
    # If headgroup is below tails, flip the entire molecule
    if head_to_tail[2] > 0:  # Positive Z means head below tail - needs flipping
        for atom in atoms_copy:
            atom['z'] = -atom['z']
        print(f"   Flipped {lipid_type} for proper upper leaflet orientation")
    
    return atoms_copy

def orient_lipid_lower_leaflet(atoms, lipid_type):
    """
    Orient lipid for lower leaflet positioning (headgroup down, tails up).
    
    ORIENTATION STRATEGY:
    • Start with upper leaflet orientation
    • Apply symmetric transformation: (x,y,z) → (x,-y,-z)
    • Results in headgroups toward negative Z (lower solution interface)
    • Tails toward positive Z (membrane center)
    
    This transformation maintains proper membrane symmetry while
    ensuring both leaflets have correct polarity.
    
    Args:
        atoms (list): List of atom dictionaries
        lipid_type (str): Lipid type identifier
        
    Returns:
        list: Oriented atom list for lower leaflet
    """
    # First orient as upper leaflet (standardizes initial orientation)
    atoms_copy = orient_lipid_upper_leaflet(atoms, lipid_type)
    
    # Apply symmetric transformation for lower leaflet
    # This maintains membrane symmetry while reversing polarity
    for atom in atoms_copy:
        atom['y'] = -atom['y']  # Reflect across X-Z plane
        atom['z'] = -atom['z']  # Invert Z-direction
    
    return atoms_copy

# ============================================================================
# MEMBRANE GEOMETRY AND PARAMETERS
# ============================================================================

def calculate_membrane_parameters(lipid_numbers, apl):
    """
    Calculate optimal membrane geometry parameters for stable assembly.
    
    MEMBRANE DESIGN SPECIFICATIONS:
    
    • Box Dimensions: Square XY plane sized for lipid area requirements
    • Membrane Width: 55Å total width accommodates lysine extension
    • Headgroup Separation: 40Å P1-P1 distance between leaflets
    • Z-Windows: Flexible positioning windows for PackMol convergence
    • Leaflet Targets: ±20Å positioning for symmetric membrane
    
    DESIGN RATIONALE:
    - 40Å P1-P1 distance provides proper membrane thickness
    - 55Å width accommodates extended lysine groups in S5LG
    - Flexible windows (±18Å) allow PackMol optimization
    - Symmetric positioning ensures balanced membrane structure
    
    Args:
        lipid_numbers (list): Number of each lipid type per leaflet
        apl (float): Area per lipid in Ų
        
    Returns:
        dict: Membrane parameters for PackMol input generation
    """
    total_lipids = sum(lipid_numbers)
    total_area = total_lipids * apl
    box_xy = math.sqrt(total_area)  # Square simulation box
    
    # MEMBRANE ARCHITECTURE PARAMETERS
    return {
        # Simulation box dimensions
        'box_xy': box_xy,                    # XY box size (calculated from lipid area)
        'box_z': 150.0,                      # Large Z dimension for assembly space
        
        # Membrane structural parameters  
        'membrane_width': 55.0,              # Total membrane width (accommodates lysine)
        'headgroup_separation': 40.0,        # Target P1-P1 distance between leaflets
        
        # Leaflet positioning targets (symmetric about membrane center)
        'upper_target': +20.0,               # Upper leaflet COM target (+40/2)
        'lower_target': -20.0,               # Lower leaflet COM target (-40/2)
        
        # PackMol assembly parameters
        'z_window': 15.0,                    # Flexible Z-positioning window (±18Å)
        'xy_margin': 0.0,                    # XY boundary margins for edge effects
        'tolerance': 1,                    # PackMol distance tolerance
        'max_cycles': 30000,                # Maximum optimization cycles
        
        # Rotation constraints (degrees)
        'rotation_xy': 10.0,                 # Limited XY rotation for stability

        'rotation_z': 360.0                  # Full Z rotation for random orientation
    }

# ============================================================================
# PACKMOL INPUT FILE GENERATION
# ============================================================================

def generate_packmol_input(pdb_files, lipid_types, lipid_numbers, analyses, params):
    """
    Generate comprehensive PACKMOL input file for membrane assembly.
    
    PACKMOL STRATEGY:
    
    1. POSITIONING CALCULATION:
       - Calculate COM targets to achieve exact headgroup alignment
       - Account for headgroup offset from COM in each lipid
       - Ensure P1-P1 distance = 40Å between leaflets
    
    2. CONSTRAINT GENERATION:
       - Create separate constraints for upper (Chain A) and lower (Chain B) leaflets
       - Apply flexible Z-windows for convergence
       - Include rotation constraints for membrane stability
    
    3. OPTIMIZATION PARAMETERS:
       - Balanced tolerance for convergence vs. precision
       - Sufficient cycles for complex membrane assembly
       - Progress monitoring for troubleshooting
    
    Args:
        pdb_files (list): Input PDB filenames
        lipid_types (list): Lipid type identifiers
        lipid_numbers (list): Lipid counts per leaflet
        analyses (list): Structural analysis results
        params (dict): Membrane geometry parameters
    """
    
    with open("PackMol.inp", 'w') as f:
        # ====================================================================
        # HEADER SECTION: Documentation and overview
        # ====================================================================
        f.write("# ================================================================\n")
        f.write("# ENHANCED MEMBRANE BUILDER - PACKMOL INPUT\n")
        f.write("# ================================================================\n")
        f.write("#\n")
        f.write("# MEMBRANE DESIGN SPECIFICATIONS:\n")
        f.write(f"# • Headgroup Alignment: P1-P1 distance = {params['headgroup_separation']:.1f}Å\n")
        f.write(f"# • Membrane Width: {params['membrane_width']:.1f}Å (accommodates lysine extension)\n")
        f.write(f"# • Chain Labels: Upper leaflet = A, Lower leaflet = B\n")
        f.write(f"# • Box Dimensions: {params['box_xy']:.1f} × {params['box_xy']:.1f} × {params['box_z']:.1f} Å\n")
        f.write(f"# • Total Lipids: {sum(lipid_numbers)} per leaflet ({sum(lipid_numbers)*2} total)\n")
        f.write("#\n")
        f.write("# SUPPORTED LIPID TYPES:\n")
        for pdb, lipid_type, count in zip(pdb_files, lipid_types, lipid_numbers):
            f.write(f"# • {lipid_type}: {count} per leaflet ({pdb})\n")
        f.write("#\n")
        f.write("# ================================================================\n")
        f.write("\n")
        
        # ====================================================================
        # PACKMOL OPTIMIZATION PARAMETERS
        # ====================================================================
        f.write("# PACKMOL OPTIMIZATION SETTINGS\n")
        f.write("# These parameters balance convergence speed with final precision\n")
        f.write(f"tolerance {params['tolerance']:.1f}        # Minimum distance between atoms (Å)\n")
        f.write("filetype pdb                      # Output format\n")
        f.write("output membrane.pdb               # Output filename\n")
        f.write("add_amber_ter                     # Add TER records for MD compatibility\n")
        f.write("seed -1                           # Random seed (-1 = time-based)\n")
        f.write(f"nloop0 {params['max_cycles']}               # Maximum optimization cycles\n")
        f.write("writeout 500                      # Progress output frequency\n")
        f.write("iprint1 10000                     # Detailed progress interval\n")
        f.write("iprint2 50000                     # Summary progress interval\n")
        f.write("fbins 4.0                         # Spatial binning for optimization\n")
        f.write("precision 0.05                    # Convergence precision\n")
        f.write("randominitialpoint                # Random initial placement\n")
        f.write("movebadrandom                     # Random repositioning for bad contacts\n")
        f.write("\n")
        
        # ====================================================================
        # HEADGROUP ALIGNMENT CALCULATIONS
        # ====================================================================
        f.write("# ================================================================\n")
        f.write("# HEADGROUP ALIGNMENT CALCULATIONS\n")
        f.write("# ================================================================\n")
        f.write("#\n")
        f.write("# Strategy: Position lipid COMs so headgroups align at target planes\n")
        f.write("# This ensures consistent P1-P1 distances across all lipid types\n")
        f.write("#\n")
        
        com_targets = {}
        print("\n🎯 Calculating precise positions for headgroup alignment:")
        print(f"   Target P1-P1 separation: {params['headgroup_separation']:.1f}Å")
        print(f"   Upper leaflet target: +{abs(params['upper_target']):.1f}Å")
        print(f"   Lower leaflet target: {params['lower_target']:.1f}Å")
        print()
        
        for lipid_type, analysis in zip(lipid_types, analyses):
            if analysis:
                # Calculate headgroup offset from COM
                head_offset = analysis['headgroup_center'] - analysis['com_z']
                
                # Position COM to place headgroups at target planes
                # Upper leaflet: headgroup at +20Å
                # Lower leaflet: headgroup at -20Å (after flipping)
                upper_com = params['upper_target'] - head_offset
                lower_com = params['lower_target'] - head_offset
                
                com_targets[lipid_type] = {
                    'upper': upper_com,
                    'lower': lower_com,
                    'head_offset': head_offset
                }
                
                # Calculate resulting headgroup positions for verification
                upper_headgroup_pos = upper_com + head_offset
                lower_headgroup_pos = lower_com + (-head_offset)  # Flipped for lower leaflet
                
                # Document calculations in PackMol file
                f.write(f"# {lipid_type} POSITIONING:\n")
                f.write(f"#   Headgroup offset from COM: {head_offset:+.1f}Å\n")
                f.write(f"#   Upper COM target: {upper_com:+.1f}Å → Headgroup at {upper_headgroup_pos:+.1f}Å\n")
                f.write(f"#   Lower COM target: {lower_com:+.1f}Å → Headgroup at {lower_headgroup_pos:+.1f}Å\n")
                
                # Calculate and verify distances
                if lipid_type in ["S5PG", "S5LG", "S5CL"]:
                    p1_distance = upper_headgroup_pos - lower_headgroup_pos
                    f.write(f"#   → P1-P1 distance: {p1_distance:.1f}Å\n")
                    print(f"   {lipid_type}: P1-P1 distance = {p1_distance:.1f}Å")
                else:
                    sugar_distance = upper_headgroup_pos - lower_headgroup_pos  
                    f.write(f"#   → Sugar-sugar distance: {sugar_distance:.1f}Å\n")
                    print(f"   {lipid_type}: Sugar-sugar distance = {sugar_distance:.1f}Å")
                
                # Tail contact analysis
                tail_center = analysis['tail_center']
                tail_offset = tail_center - analysis['com_z']
                upper_tail = upper_com + tail_offset
                lower_tail = lower_com + (-tail_offset)
                tail_gap = abs(upper_tail - lower_tail)
                
                f.write(f"#   → Tail contact gap: {tail_gap:.1f}Å\n")
                print(f"     Tail contact gap: {tail_gap:.1f}Å")
                
                f.write("#\n")
                
            else:
                # Fallback for missing analysis
                com_targets[lipid_type] = {
                    'upper': params['upper_target'],
                    'lower': params['lower_target'],
                    'head_offset': 0.0
                }
                f.write(f"# {lipid_type}: Using default positioning (analysis unavailable)\n")
        
        f.write("\n")
        
        # ====================================================================
        # MEMBRANE ASSEMBLY CONSTRAINTS
        # ====================================================================
        f.write("# ================================================================\n")
        f.write("# MEMBRANE ASSEMBLY CONSTRAINTS\n") 
        f.write("# ================================================================\n")
        f.write("\n")
        
        # Generate constraints for both leaflets
        for is_upper in [True, False]:
            leaflet_name = "UPPER" if is_upper else "LOWER"
            chain_id = "A" if is_upper else "B"
            
            f.write(f"# {leaflet_name} LEAFLET (CHAIN {chain_id})\n")
            f.write(f"# Headgroups face {'upward (+Z)' if is_upper else 'downward (-Z)'}\n")
            f.write(f"# Tails point toward membrane center\n")
            f.write("#\n")
            
            for pdb_file, lipid_type, count in zip(pdb_files, lipid_types, lipid_numbers):
                prefix = "upper" if is_upper else "lower"
                oriented_file = f"{prefix}_{pdb_file}"
                
                f.write(f"structure {oriented_file}\n")
                f.write(f"  number {count}\n")
                
                # Get target COM position for this lipid type
                if lipid_type in com_targets:
                    target_com = com_targets[lipid_type]['upper' if is_upper else 'lower']
                    head_offset = com_targets[lipid_type]['head_offset']
                else:
                    target_com = params['upper_target'] if is_upper else params['lower_target']
                    head_offset = 0.0
                
                # Calculate Z-constraints with flexible windows
                z_center = target_com
                z_window = params['z_window']
                z_min = z_center - z_window
                z_max = z_center + z_window
                
                # Prevent leaflet overlap while maintaining flexibility
                if is_upper:
                    z_min = max(z_min, 2.0)    # Upper leaflet minimum (avoid center)
                    z_max = min(z_max, 45.0)   # Upper leaflet maximum (reasonable limit)
                else:
                    z_min = max(z_min, -45.0)  # Lower leaflet minimum (reasonable limit)
                    z_max = min(z_max, -2.0)   # Lower leaflet maximum (avoid center)
                
                # XY constraints with edge margins
                margin = params['xy_margin']
                xy_min = margin
                xy_max = params['box_xy'] - margin
                
                # Document constraint rationale
                f.write(f"  # Target COM: {target_com:+.1f}Å (headgroup at {target_com + head_offset:+.1f}Å)\n")
                f.write(f"  # Z-window: {z_min:+.1f} to {z_max:+.1f}Å (±{z_window:.1f}Å flexibility)\n")
                f.write(f"  # XY-margins: {margin:.1f}Å from edges\n")
                
                # PackMol constraint specification
                f.write(f"  inside box {xy_min:.1f} {xy_min:.1f} {z_min:.1f} ")
                f.write(f"{xy_max:.1f} {xy_max:.1f} {z_max:.1f}\n")
                
                # Rotation constraints for membrane stability
                f.write(f"  # Rotation limits: XY=±{params['rotation_xy']:.0f}° (membrane stability), Z=full (random orientation)\n")
                f.write(f"  constrain_rotation x 0. {params['rotation_xy']:.0f}.\n")
                f.write(f"  constrain_rotation y 0. {params['rotation_xy']:.0f}.\n")
                f.write(f"  constrain_rotation z 0. {params['rotation_z']:.0f}.\n")
                
                # PackMol positioning options
                f.write("  centerofmass                      # Use COM for positioning\n")
                f.write("  changechains                      # Allow chain ID updates\n")
                f.write("end structure\n")
                f.write("\n")
        
        f.write("# ================================================================\n")
        f.write("# END OF PACKMOL INPUT\n")
        f.write("# ================================================================\n")

# ============================================================================
# MAIN EXECUTION WORKFLOW
# ============================================================================

def main():
    """
    Main execution workflow for enhanced membrane builder.
    
    WORKFLOW STEPS:
    1. Parse and validate command-line arguments
    2. Scale lipid numbers to target total
    3. Calculate optimal membrane parameters
    4. Process each lipid structure:
       - Identify lipid type and analyze structure
       - Create oriented versions for upper/lower leaflets
       - Apply proper chain labeling (A=upper, B=lower)
    5. Generate comprehensive PackMol input
    6. Provide assembly instructions and verification
    """
    args = parse_arguments()
    
    # ========================================================================
    # INPUT VALIDATION
    # ========================================================================
    if len(args.pdb) != len(args.numbers):
        print("❌ ERROR: Number of PDB files must match number of lipid counts!")
        print(f"   PDB files: {len(args.pdb)}")
        print(f"   Numbers: {len(args.numbers)}")
        sys.exit(1)
    
    print("=" * 80)
    print("🔬 ENHANCED MEMBRANE BUILDER WITH CHAIN LABELING")
    print("=" * 80)
    print("Building touching-tail membrane with precise headgroup alignment")
    print("Upper leaflet = Chain A, Lower leaflet = Chain B")
    print()
    
    # ========================================================================
    # LIPID COMPOSITION SCALING
    # ========================================================================
    print("📊 Optimizing lipid composition:")
    target_total = args.target_total
    current_total = sum(args.numbers)
    
    if current_total != target_total:
        print(f"   Scaling from {current_total} to {target_total} lipids per leaflet")
        
        # Proportional scaling with integer constraints
        scale_factor = target_total / current_total
        scaled_numbers = [max(1, int(n * scale_factor)) for n in args.numbers]
        
        # Fine-tune to exact total
        diff = target_total - sum(scaled_numbers)
        while diff > 0:
            # Add to largest component
            max_idx = scaled_numbers.index(max(scaled_numbers))
            scaled_numbers[max_idx] += 1
            diff -= 1
        while diff < 0:
            # Remove from largest component (if > 1)
            max_vals = [i for i, x in enumerate(scaled_numbers) if x == max(scaled_numbers)]
            for idx in max_vals:
                if scaled_numbers[idx] > 1:
                    scaled_numbers[idx] -= 1
                    break
            diff += 1
    else:
        scaled_numbers = args.numbers[:]
    
    # Display final composition
    print("   Final composition per leaflet:")
    for pdb, orig, scaled in zip(args.pdb, args.numbers, scaled_numbers):
        if orig != scaled:
            print(f"     {pdb}: {orig} → {scaled}")
        else:
            print(f"     {pdb}: {scaled}")
    print(f"   Total per leaflet: {sum(scaled_numbers)} (×2 = {sum(scaled_numbers)*2} total)")
    print()
    
    # ========================================================================
    # MEMBRANE PARAMETER CALCULATION
    # ========================================================================
    params = calculate_membrane_parameters(scaled_numbers, args.apl)
    
    print("📐 Membrane architecture design:")
    print(f"   Simulation box: {params['box_xy']:.1f} × {params['box_xy']:.1f} × {params['box_z']:.1f} Å")
    print(f"   Area per lipid: {args.apl:.1f} Ų")
    print(f"   Total membrane area: {sum(scaled_numbers) * args.apl:.0f} Ų")
    print(f"   Membrane width: {params['membrane_width']:.1f} Å")
    print(f"   P1-P1 separation: {params['headgroup_separation']:.1f} Å")
    print(f"   Leaflet targets: ±{abs(params['upper_target']):.1f} Å")
    print(f"   Positioning windows: ±{params['z_window']:.1f} Å")
    print(f"   PackMol tolerance: {params['tolerance']:.1f} Å")
    print()
    
    # ========================================================================
    # LIPID STRUCTURE PROCESSING
    # ========================================================================
    lipid_types = []
    analyses = []
    
    print("🧬 Processing lipid structures and creating oriented versions:")
    for i, (pdb_file, count) in enumerate(zip(args.pdb, scaled_numbers)):
        print(f"\n📄 {pdb_file} ({count} per leaflet)")
        
        # Automatic lipid type identification
        lipid_type = "UNKNOWN"
        for lt in ["S5PG", "S5LG", "S5DG", "S5CL"]:
            if lt in pdb_file.upper():
                lipid_type = lt
                break
        
        lipid_types.append(lipid_type)
        print(f"   🏷️  Type: {lipid_type}")
        
        # Read and validate PDB structure
        atoms = read_pdb_atoms(pdb_file)
        if not atoms:
            print(f"   ❌ ERROR: No atoms found in {pdb_file}!")
            sys.exit(1)
        
        print(f"   📊 Atoms: {len(atoms)}")
        
        # Perform structural analysis
        print(f"   🔍 Analyzing structure...")
        analysis = analyze_lipid_structure(atoms, lipid_type)
        analyses.append(analysis)
        
        if analysis:
            print(f"     Total span: {analysis['total_span']:.1f} Å")
            print(f"     Headgroup region: {analysis['headgroup_min']:.1f} to {analysis['headgroup_max']:.1f} Å")
            print(f"     Tail region: {analysis['tail_min']:.1f} to {analysis['tail_max']:.1f} Å")
            print(f"     Head-tail vector: {analysis['head_to_tail_vector']:+.1f} Å")
        else:
            print(f"     ⚠️  Structure analysis failed - using defaults")
        
        # Preserve original header information
        header_lines = get_pdb_header(pdb_file)
        
        # Create oriented versions for membrane assembly
        print(f"   🔄 Creating oriented structures...")
        
        # Upper leaflet: Chain A
        upper_atoms = orient_lipid_upper_leaflet(atoms, lipid_type)
        upper_file = f"upper_{pdb_file}"
        write_pdb_atoms(upper_atoms, upper_file, header_lines, chain_id='A')
        print(f"     Upper leaflet (Chain A): {upper_file}")
        
        # Lower leaflet: Chain B  
        lower_atoms = orient_lipid_lower_leaflet(atoms, lipid_type)
        lower_file = f"lower_{pdb_file}"
        write_pdb_atoms(lower_atoms, lower_file, header_lines, chain_id='B')
        print(f"     Lower leaflet (Chain B): {lower_file}")
    
    # ========================================================================
    # PACKMOL INPUT GENERATION
    # ========================================================================
    print("\n📝 Generating comprehensive PACKMOL input...")
    generate_packmol_input(args.pdb, lipid_types, scaled_numbers, analyses, params)
    
    print("✅ PackMol.inp generated successfully!")
    
    # ========================================================================
    # ASSEMBLY INSTRUCTIONS AND VERIFICATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("🎯 MEMBRANE ASSEMBLY SUMMARY")
    print("=" * 80)
    print()
    print("📋 DESIGN SPECIFICATIONS:")
    print(f"   ✓ Headgroup alignment: P1-P1 distance = {params['headgroup_separation']:.1f}Å")
    print(f"   ✓ Chain labeling: Upper = A, Lower = B")
    print(f"   ✓ Membrane width: {params['membrane_width']:.1f}Å (lysine compatible)")
    print(f"   ✓ Touching tails: No interdigitation")
    print(f"   ✓ Symmetric structure: ±{abs(params['upper_target']):.1f}Å leaflet separation")
    print()
    
    print("📂 GENERATED FILES:")
    for pdb_file in args.pdb:
        print(f"   • upper_{pdb_file} (Chain A - Upper leaflet)")
        print(f"   • lower_{pdb_file} (Chain B - Lower leaflet)")
    print(f"   • PackMol.inp (Assembly instructions)")
    print()
    
    print("🚀 ASSEMBLY INSTRUCTIONS:")
    print("   1. Ensure PACKMOL is installed and accessible")
    print("   2. Run: packmol < PackMol.inp")
    print("   3. Output: membrane.pdb with chain-labeled leaflets")
    print("   4. Verify structure in VMD or similar visualization tool")
    print()
    
    print("🔍 VERIFICATION CHECKLIST:")
    print("   □ Check P1-P1 distances between leaflets (~40Å)")
    print("   □ Verify chain labels (A=upper, B=lower)")
    print("   □ Confirm headgroup alignment across lipid types")
    print("   □ Ensure tail contact without overlap")
    print("   □ Validate membrane symmetry")
    print()
    
    print("💡 TROUBLESHOOTING:")
    print("   • If convergence issues: Increase tolerance or z_window")
    print("   • If overlap problems: Adjust xy_margin or membrane_width")
    print("   • If alignment issues: Check lipid orientation files")
    print("   • For debugging: Enable PackMol verbose output")
    
    print("=" * 80)

# ============================================================================
# PROGRAM ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Program entry point with comprehensive error handling.
    
    Handles user interruption and provides detailed error reporting
    for troubleshooting membrane assembly issues.
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Assembly interrupted by user")
        print("   Partial files may have been created")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        print("\n🔧 Debug information:")
        import traceback
        traceback.print_exc()
        print("\n💡 Please check:")
        print("   • Input PDB file format and accessibility")
        print("   • Command-line argument validity")
        print("   • Available disk space for output files")
        print("   • Python package dependencies")
        sys.exit(1)
