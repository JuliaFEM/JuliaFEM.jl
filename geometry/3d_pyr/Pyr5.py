# -*- coding: utf-8 -*-

###
### This file is generated automatically by SALOME v7.8.0 with dump python functionality
###

import sys
import salome

salome.salome_init()
theStudy = salome.myStudy

import salome_notebook
notebook = salome_notebook.NoteBook(theStudy)

###
### GEOM component
###

import GEOM
from salome.geom import geomBuilder
import math
import SALOMEDS


geompy = geomBuilder.New(theStudy)

O = geompy.MakeVertex(0, 0, 0)
OX = geompy.MakeVectorDXDYDZ(1, 0, 0)
OY = geompy.MakeVectorDXDYDZ(0, 1, 0)
OZ = geompy.MakeVectorDXDYDZ(0, 0, 1)
geompy.addToStudy( O, 'O' )
geompy.addToStudy( OX, 'OX' )
geompy.addToStudy( OY, 'OY' )
geompy.addToStudy( OZ, 'OZ' )

###
### SMESH component
###

import  SMESH, SALOMEDS
from salome.smesh import smeshBuilder

smesh = smeshBuilder.New(theStudy)
Pyr5 = smesh.Mesh()
coords = [
         [-1.0,-1.0,-1.0], # N1
         [ 1.0,-1.0,-1.0], # N2
         [ 1.0, 1.0,-1.0], # N3
         [-1.0, 1.0,-1.0], # N4
         [ 0.0, 0.0, 1.0]] # N5
for c in coords:
	Pyr5.AddNode( *c )

volID = Pyr5.AddVolume( [ 1, 4, 3, 2, 5 ] ) # http://docs.salome-platform.org/latest/gui/SMESH/connectivity_page.html
Pyr5_1 = Pyr5.CreateEmptyGroup( SMESH.VOLUME, 'Pyr5' )
nbAdd = Pyr5_1.AddFrom( Pyr5.GetMesh() )
nbAdded, Pyr5, _NoneGroup = Pyr5.MakeBoundaryElements( SMESH.BND_2DFROM3D, '', '', 0, [])
baseQuad = Pyr5.CreateEmptyGroup( SMESH.FACE, 'baseQuad' )
nbAdd = baseQuad.Add( [ 2 ] )
tipPoint = Pyr5.CreateEmptyGroup( SMESH.NODE, 'tipPoint' )
nbAdd = tipPoint.Add( [ 5 ] )


## Set names of Mesh objects
smesh.SetName(baseQuad, 'baseQuad')
smesh.SetName(Pyr5.GetMesh(), 'Pyr5')
smesh.SetName(Pyr5_1, 'Pyr5')
smesh.SetName(tipPoint, 'tipPoint')


if salome.sg.hasDesktop():
  salome.sg.updateObjBrowser(1)
