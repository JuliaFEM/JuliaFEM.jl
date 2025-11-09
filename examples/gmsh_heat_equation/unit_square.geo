// Gmsh geometry file: Unit square mesh for heat equation tutorial
// Generate with: gmsh -2 unit_square.geo -o unit_square.msh

// Mesh element size
lc = 0.1;

// Corner points
Point(1) = {0, 0, 0, lc};
Point(2) = {1, 0, 0, lc};
Point(3) = {1, 1, 0, lc};
Point(4) = {0, 1, 0, lc};

// Edges
Line(1) = {1, 2};  // Bottom
Line(2) = {2, 3};  // Right
Line(3) = {3, 4};  // Top
Line(4) = {4, 1};  // Left

// Surface
Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Physical groups for boundary conditions
Physical Line("bottom") = {1};
Physical Line("right") = {2};
Physical Line("top") = {3};
Physical Line("left") = {4};
Physical Surface("body") = {1};

// Use triangular elements
Mesh.ElementOrder = 1;  // Linear elements (Tri3)
Mesh.Algorithm = 6;     // Frontal-Delaunay for 2D
