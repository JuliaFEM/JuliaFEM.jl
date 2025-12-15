# I/O Module - Mesh Import/Export

**Purpose:** Read and write mesh files from various FEM software packages.

## Supported Formats

### Abaqus (.inp)
**Status:** ✅ Active  
**Files:**
- `AbaqusReader.jl` - Main reader with keyword parsing
- `abaqus_reader.jl` - Simplified reader
- `abaqus_download.jl` - Download example meshes
- `keyword_register.jl` - Keyword parsing system

**Use:** `mesh = read_abaqus("model.inp")`

### Code Aster (.med, .rmed)
**Status:** ⚠️ Requires HDF5 (weak dependency)  
**Files:**
- `AsterReader.jl` - Main Aster reader
- `aster_reader.jl` - Simplified reader
- `read_aster_mesh.jl` - HDF5-based mesh reading (NOT included - needs HDF5.jl)
- `read_aster_results.jl` - HDF5-based results reading (NOT included)

**Strategy:** Implement as weak dependency - only load if HDF5.jl available

### Gmsh (.msh)
**Status:** ✅ Active  
**Files:**
- `gmsh_reader.jl` - Gmsh mesh file parser

**Use:** `mesh = read_gmsh("mesh.msh")`

## File Organization

### Core Readers
- `io.jl` - Common I/O utilities
- `gmsh_reader.jl` - Gmsh format
- `abaqus_reader.jl` - Simple Abaqus reader
- `aster_reader.jl` - Simple Aster reader (no HDF5)

### Legacy Vendor Package Infrastructure
- `AbaqusReader.jl` - Full Abaqus reader (from vendor package)
- `AsterReader.jl` - Full Aster reader (from vendor package)
- `keyword_register.jl` - Keyword parsing system
- `parse_mesh.jl` - Generic mesh parsing
- `parse_model.jl` - Model structure parsing
- `create_surface_elements.jl` - Surface element extraction

### Disabled (Require HDF5)
- `read_aster_mesh.jl` - ❌ Commented out (needs HDF5.jl)
- `read_aster_results.jl` - ❌ Commented out (needs HDF5.jl)

## Design Philosophy

### Weak Dependencies
For optional formats requiring heavy dependencies (HDF5):
```julia
# In Project.toml
[extras]
HDF5 = "..."

# In src/io/
if isdefined(Main, :HDF5)
    include("read_aster_mesh.jl")
end
```

### Two-Tier Strategy
1. **Simple readers** - Basic functionality, minimal dependencies
2. **Full readers** - Complete keyword support, complex features

Users can choose based on needs.

## Future Work

### Export Formats
- VTK/VTU for visualization
- Exodus II for multi-physics
- JSON for web applications

### Import Enhancements
- NASTRAN (.bdf, .nas)
- ANSYS (.cdb)
- CalculiX (.inp)

### Consolidation
- Unify simple vs full reader approaches
- Document which reader to use when
- Benchmarks for large meshes

## Usage Examples

```julia
using JuliaFEM

# Abaqus
mesh = read_abaqus("cantilever.inp")

# Gmsh (if gmsh_jll available)
mesh = read_gmsh("geometry.msh")

# Aster (if HDF5 available)
mesh = read_aster("model.med")
```

## Dependencies

**Required:**
- None (pure Julia)

**Optional:**
- `gmsh_jll` - For Gmsh API access
- `HDF5.jl` - For Code Aster .med files
- `Downloads.jl` - For downloading example meshes

---

**Maintainer:** JuliaFEM Team  
**Last Updated:** November 21, 2025
