# Enhanced Membrane Builder

![Python](https://img.shields.io/badge/python-3.6+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

A comprehensive Python toolkit for building realistic interdigitated lipid membranes using PACKMOL and MDAnalysis. Designed specifically for ATB-generated phospholipids and glycolipids with optimized parameters for fast convergence.

![Membrane Structure](https://github.com/yourusername/membrane-builder/raw/main/membrane_example.png)
*525 lipids per leaflet membrane system built with this toolkit*

## 🚀 Quick Start

```bash
# 1. Orient lipids along Z-axis
python3 orient_lipids.py

# 2. Generate PACKMOL membrane (separated leaflets)
python3 genPackMol.py --pdb oriented_02_S5PG.pdb oriented_02_S5LG.pdb oriented_02_S5DG.pdb oriented_02_S5CL.pdb \
                     --numbers 56 12 19 13 --apl 60 --target_total 525

# 3. Run PACKMOL assembly
packmol < PackMol.inp

# 4. Compact leaflets to touching distance
python3 compactMembrane.py

# 5. Solvate with GROMACS
gmx solvate -cp membrane_compacted.pdb -o solv.pdb -cs spc216 -p membrane.top

# 6. Remove water between leaflets
python3 dryMem.py
```

## 📋 Complete Workflow

### Step 1: `orient_lipids.py` - Lipid Pre-orientation

**Purpose**: Aligns lipid molecules along the Z-axis for optimal PACKMOL packing.

**Requirements**: Input PDB files must be named exactly as hardcoded in the script:
- `02_S5PG.pdb` (Phosphatidylglycerol)
- `02_S5DG.pdb` (Digalactosyldiacylglycerol) 
- `02_S5CL.pdb` (Cardiolipin)
- `02_S5LG.pdb` (Lysylphosphatidylglycerol)

**Function**: 
- Identifies functional groups (headgroup, glycerol, tails) using ATB naming conventions
- Orients molecules with headgroups up (+Z), glycerol at center (Z=0), tails down (-Z)
- Uses Rodrigues' rotation formula for precise 3D orientation
- Essential for proper membrane assembly

**Outputs**: `oriented_*.pdb` files ready for membrane building

### Step 2: `genPackMol.py` - PACKMOL Membrane Assembly

**Purpose**: Generates PACKMOL input for building separated-leaflet membranes optimized for fast convergence.

**Usage**:
```bash
python3 genPackMol.py --pdb oriented_02_S5PG.pdb oriented_02_S5LG.pdb oriented_02_S5DG.pdb oriented_02_S5CL.pdb \
                     --numbers 56 12 19 13 --apl 60 --target_total 525
```

**Key Parameters**:
- `--pdb`: Oriented lipid PDB files
- `--numbers`: Lipid counts (automatically scaled to target_total)
- `--apl`: Area per lipid (Ų) - determines box XY dimensions via `sqrt(apl × total_lipids)`
- `--target_total`: Target lipids per leaflet

**Critical Design Features**:
- **Leaflet Separation**: Places leaflets 20-40Å apart for PACKMOL convergence
- **Z-Window Parameter** (`z_window = 15.0`): Provides ±15Å flexibility around target positions
  - Enables realistic timeframe convergence (few minutes for 525 lipids/leaflet)
  - Prevents immediate interdigitation that would cause convergence failure
- **Optimization Steps**: Set to 30,000 cycles (`max_cycles`) for balance of speed/quality
- **Chain Labeling**: Upper leaflet = Chain A, Lower leaflet = Chain B

**Customizable Settings** (in `calculate_membrane_parameters()`):
```python
'tolerance': 1.0,              # PACKMOL distance tolerance (Å)
'max_cycles': 30000,           # Optimization cycles
'z_window': 15.0,              # Z-positioning flexibility (±Å)
'rotation_xy': 10.0,           # XY rotation constraint (degrees)
'headgroup_separation': 40.0   # Target P1-P1 distance (Å)
```

**Output**: `PackMol.inp` and oriented leaflet PDB files

### Step 3: `compactMembrane.py` - Leaflet Compaction

**Purpose**: Moves separated leaflets to touching distance and centers membrane in simulation box.

**Function**:
- Iteratively moves upper leaflet down 0.3Å, lower leaflet up 0.3Å per cycle
- Stops when minimum inter-leaflet distance reaches 2.6Å
- Translates entire system up 75Å to center in box
- Assigns final chain IDs (A=upper, B=lower)

**Critical Box Parameters** (must be edited manually):
```python
box_xy = 177.4823934930  # sqrt(525 × 60) for your system
box_z = 150.0000000000   # Standard membrane box height
```

**For Different Systems**: Update `box_xy = sqrt(n_lipids × apl)` where:
- `n_lipids` = lipids per leaflet
- `apl` = area per lipid used in Step 2

**Performance**: 
- Uses sampling for large systems (>100k atom pairs)
- Vectorized operations for speed
- Memory-efficient distance calculations

**Inputs**: `membrane.pdb` (from PACKMOL)
**Outputs**: `membrane_compacted.pdb` (touching leaflets, centered, chain-labeled)

### Step 4: `dryMem.py` - Inter-leaflet Water Removal

**Purpose**: Removes water molecules between membrane leaflets after solvation.

**Prerequisites**: 
```bash
gmx solvate -cp membrane_compacted.pdb -o solv.pdb -cs spc216 -p membrane.top
```

**Algorithm**:
1. Identifies membrane normal axis (largest P1/P2 COM difference)
2. Defines inter-leaflet region between P1/P2 centers of mass
3. Removes water molecules that are:
   - Located between leaflets AND
   - Within 8Å of any membrane atom
4. Uses memory-efficient chunked processing for large systems

**Key Parameters**:
```python
cutoff_distance = 8.0  # Å from membrane atoms
chunk_size = 1000      # Water molecules per processing chunk
```

**For Different Membranes**:
- Automatically detects membrane orientation
- Cutoff distance may need adjustment for different lipid compositions
- Chunk size can be increased for systems with more RAM

**Inputs**: `solv.pdb` (solvated membrane)
**Outputs**: `solv_drymem.pdb` (dry inter-leaflet region)

## 🎛️ System Customization

### Membrane Composition
Modify lipid numbers and types in Step 2. Supported ATB lipids:
- **S5PG**: Phosphatidylglycerol 
- **S5LG**: Lysylphosphatidylglycerol
- **S5DG**: Digalactosyldiacylglycerol  
- **S5CL**: Cardiolipin

### System Size
1. Adjust `--target_total` in Step 2
2. Update `box_xy` calculation in Step 3: `sqrt(target_total × apl)`
3. Ensure adequate RAM for large systems

### Membrane Properties
- **Density**: Modify `--apl` (typical range: 50-80 Ų)
- **Thickness**: Adjust `headgroup_separation` in `genPackMol.py`
- **Packing Speed**: Increase `z_window` for faster convergence

## 🔧 Troubleshooting

**PACKMOL convergence issues**:
- Increase `tolerance` to 2.0-3.0
- Increase `z_window` to 20-25Å  
- Reduce `target_total` for testing

**Memory issues in compaction**:
- Reduce `sample_size` in `calculate_max_distance_fast()`
- Use precise mode for small systems (<10k atoms)

**Water removal too aggressive**:
- Reduce cutoff distance from 8Å to 5-6Å
- Increase chunk size if RAM permits

## 🧪 Technical Details

**PACKMOL Optimization Strategy**:
- Balanced tolerance (1.0Å) for speed vs precision
- Flexible Z-windows prevent immediate interdigitation
- Constrained rotations maintain membrane stability
- Progress monitoring every 500 cycles

**Distance Calculations**:
- Sampling approach for >100k atom pairs
- Exact calculations for smaller systems  
- Vectorized operations using NumPy/SciPy

**Memory Management**:
- Chunked processing for water analysis
- Efficient MDAnalysis selections
- Copy-on-write for large coordinate arrays

## 📚 Dependencies

```bash
pip install numpy scipy MDAnalysis
```

**External Requirements**:
- PACKMOL (molecular packing)
- GROMACS (solvation)
- VMD (visualization, optional)

---
*Optimized for ATB force fields and realistic membrane simulations. Currently only builds for ATB lipid naming with glycolipids and phospholipids*
