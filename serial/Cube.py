from PyQuante.Constants import bohr2ang,ang2bohr
from PyQuante import Molecule
import numpy as np
import ase.io.cube
import os
import sys
"""
 Code for reading/writing Gaussian Cube files
 
 This program is part of the PyQuante quantum chemistry program suite.

 Copyright (c) 2004, Richard P. Muller. All Rights Reserved. 

 PyQuante version 1.2 and later is covered by the modified BSD
 license. Please see the file LICENSE that is part of this
 distribution. 
"""

####################################################
#           Get the size of the box
####################################################
def get_bbox(atoms,**kwargs):
    dbuff = kwargs.get('dbuff',2.5)
    big = kwargs.get('big',10000)
    xmin = ymin = zmin = big
    xmax = ymax = zmax = -big

    # get the limit of the atoms
    for atom in atoms:
        x,y,z = atom.pos()
        xmin = min(xmin,x)
        ymin = min(ymin,y)
        zmin = min(zmin,z)
        xmax = max(xmax,x)
        ymax = max(ymax,y)
        zmax = max(zmax,z)

    # add the buffers
    xmin -= dbuff
    ymin -= dbuff
    zmin -= dbuff
    xmax += dbuff
    ymax += dbuff
    zmax += dbuff

    # round to 0.5 Ang
    xmin = np.sign(xmin)*np.ceil(2*np.abs(xmin))/2
    ymin = np.sign(ymin)*np.ceil(2*np.abs(ymin))/2
    zmin = np.sign(zmin)*np.ceil(2*np.abs(zmin))/2
    xmax = np.sign(xmax)*np.ceil(2*np.abs(xmax))/2
    ymax = np.sign(ymax)*np.ceil(2*np.abs(ymax))/2
    zmax = np.sign(zmax)*np.ceil(2*np.abs(zmax))/2

    return (xmin,xmax),(ymin,ymax),(zmin,zmax)

####################################################
#         Mesh individual orbitals
####################################################
def mesh_orb(file_name,atoms,bfs,orbs,index):

    
    (xmin,xmax),(ymin,ymax),(zmin,zmax) = get_bbox(atoms)
    dx,dy,dz = xmax-xmin,ymax-ymin,zmax-zmin
    ppb = 2.0 # Points per bohr # well nope per angs.
    spacing = 1.0/ppb
    nx,ny,nz = int(dx*ppb)+1,int(dy*ppb)+1,int(dz*ppb)+1

    
    print("\t\tWriting Gaussian Cube file {}".format(file_name))

    f = open(file_name,'w')
    f.write("CUBE FILE\n")
    f.write("OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")
    f.write("%5i %11.6f %11.6f %11.6f\n" %  (len(atoms),xmin,ymin,zmin))
    f.write("%5i %11.6f %11.6f %11.6f\n" %  (nx,spacing,0,0))
    f.write("%5i %11.6f %11.6f %11.6f\n" %  (ny,0,spacing,0))
    f.write("%5i %11.6f %11.6f %11.6f\n" %  (nz,0,0,spacing))

    # The second record here is the nuclear charge, which differs from the
    #  atomic number when a ppot is used. Since I don't have that info, I'll
    #  just echo the atno
    for atom in atoms:
        atno = atom.atno
        x,y,z = atom.pos()
        f.write("%5i %11.6f %11.6f %11.6f %11.6f\n" %  (atno,atno,x,y,z))

    nbf = len(bfs)
    f.write(" ")

    for i in xrange(nx):
        xg = xmin + i*spacing

        for j in xrange(ny):
            yg = ymin + j*spacing

            for k in xrange(nz):
                zg = zmin + k*spacing

                amp = 0

                for ibf in xrange(nbf):
                    amp += bfs[ibf].amp(xg,yg,zg)*orbs[ibf,index]

                if abs(amp) < 1e-12: 
                    amp = 0
                f.write(" %11.5e" % amp.real)
                if k % 6 == 5: 
                    f.write("\n")
            f.write("\n")
    f.close()
    xyz_min = np.array([xmin,ymin,zmin])
    nb_pts = np.array([nx,ny,nz])
    return file_name,xyz_min,nb_pts,spacing

####################################################
#          Create the volumetric
#           data for blender
####################################################
def cube2blender(fname):

    #print("Reading cube file {}".format(fname))
    data, atoms = ase.io.cube.read_cube_data(fname)

    # Here, I want the electron density, not the wave function
    data = data**2

    # If data is too large, just reduce it by striding with steps >1
    sx, sy, sz = 1, 1, 1
    data = data[::sx,::sy,::sz]

    # Note the reversed order!!
    nz, ny, nx = data.shape
    nframes = 1
    header = np.array([nx,ny,nz,nframes])

    #open and write to file
    #if np.max(data) != 0:
    #    vdata = data.flatten() / np.max(data)
    #else:
    #    vdata = data.flatten()
    vdata = data.flatten()
    vfname = os.path.splitext(fname)[0] + '.bvox'
    vfile = open(vfname,'wb')
    #print("\t\tWriting Blender voxel file {}".format(vfname))
    header.astype('<i4').tofile(vfile)
    vdata.astype('<f4').tofile(vfile)

    return vfname

####################################################
#         Mesh excited densities
####################################################
def mesh_exc_dens(atoms,bfs,N,orbs,time_step,nocc,resolution=2):
        
    # get the size of the box and
    # resolution all that
    (xmin,xmax),(ymin,ymax),(zmin,zmax) = get_bbox(atoms)
    dx,dy,dz = xmax-xmin,ymax-ymin,zmax-zmin
    ppb = float(resolution) # Points per bohr # well nope per angs.
    spacing = 1.0/ppb
    nx,ny,nz = int(dx*ppb)+1,int(dy*ppb)+1,int(dz*ppb)+1

    # print the cube file
    file_name_elec = atoms.name+'_elec_%04d.cube' %(time_step)
    file_name_hole = atoms.name+'_hole_%04d.cube' %(time_step)

    #print("\t\tWriting Gaussian Cube files %s %s" %(file_name_elec,file_name_hole))
    felec = open(file_name_elec,'w')
    fhole = open(file_name_hole,'w')

    felec.write("CUBE FILE\n")
    felec.write("OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")
    felec.write("%5i %11.6f %11.6f %11.6f\n" %  (len(atoms),xmin,ymin,zmin))
    felec.write("%5i %11.6f %11.6f %11.6f\n" %  (nx,spacing,0,0))
    felec.write("%5i %11.6f %11.6f %11.6f\n" %  (ny,0,spacing,0))
    felec.write("%5i %11.6f %11.6f %11.6f\n" %  (nz,0,0,spacing))

    fhole.write("CUBE FILE\n")
    fhole.write("OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")
    fhole.write("%5i %11.6f %11.6f %11.6f\n" %  (len(atoms),xmin,ymin,zmin))
    fhole.write("%5i %11.6f %11.6f %11.6f\n" %  (nx,spacing,0,0))
    fhole.write("%5i %11.6f %11.6f %11.6f\n" %  (ny,0,spacing,0))
    fhole.write("%5i %11.6f %11.6f %11.6f\n" %  (nz,0,0,spacing))

    # The second record here is the nuclear charge, which differs from the
    #  atomic number when a ppot is used. Since I don't have that info, I'll
    #  just echo the atno
    for atom in atoms:
        atno = atom.atno
        x,y,z = atom.pos()
        felec.write("%5i %11.6f %11.6f %11.6f %11.6f\n" %  (atno,atno,x,y,z))
        fhole.write("%5i %11.6f %11.6f %11.6f %11.6f\n" %  (atno,atno,x,y,z))

    nbf = len(bfs)
    felec.write(" ")
    fhole.write(" ")

    for i in xrange(nx):
        xg = xmin + i*spacing

        for j in xrange(ny):
            yg = ymin + j*spacing

            for k in xrange(nz):
                zg = zmin + k*spacing

                amp_elec = 0
                amp_hole = 0


                for index in range(nocc):
                    pop = 1.- N[index,time_step]
                    if pop > 1E-12:
                        for ibf in range(nbf): 
                            amp_hole += pop*bfs[ibf].amp(xg,yg,zg)*orbs[ibf,index]


                for index in xrange(nocc,nbf):
                    pop = N[index,time_step]
                    if pop > 1E-12:
                        for ibf in range(nbf):
                            amp_elec += pop*bfs[ibf].amp(xg,yg,zg)*orbs[ibf,index]

                if abs(amp_hole) < 1e-12: 
                    amp_hole = 0

                if abs(amp_elec) < 1e-12: 
                    amp_elec = 0

                fhole.write(" %11.5e" % amp_hole.real)
                felec.write(" %11.5e" % amp_elec.real)

                if k % 6 == 5: 
                    fhole.write("\n")
                    felec.write("\n")

            fhole.write("\n")
            felec.write("\n")

    # close the file
    felec.close()
    fhole.close()
    xyz_min = np.array([xmin,ymin,zmin])
    nb_pts = np.array([nx,ny,nz])
    return file_name_elec,file_name_hole,xyz_min,nb_pts,spacing

####################################################
#         create the VMD animation file
####################################################
def create_vmd_anim(mol_name,nT,step_mo,color_elec=18,color_hole=21,iso_elec=1E-5,iso_hole=1E-5):


    vmd_script = 'animate_voxel.vmd'
    print("\t Writing VMD Script %s " %vmd_script   )
    f = open(vmd_script,'w')

    f.write('# Display settings\n')
    f.write('mol delete top\n')
    f.write('display projection   Orthographic\n')
    f.write('display nearclip set 0.000000\n')
    f.write('display farclip  set 10.000000\n')
    f.write('display depthcue   off\n\n')


    f.write('# store the molecule id for later use\n')
    # original molecule 
    fname_up = mol_name+'_elec_%04d.cube' %(0)
    f.write('set updmol [mol new {%s} type cube waitfor all]\n' %(fname_up))
    for iT in range(0,nT,step_mo):
        fname = mol_name+'_elec_%04d.cube' %(iT)
        f.write('mol addfile {%s} type cube waitfor all\n' %fname)
    f.write('\n')

    for iT in range(0,nT,step_mo):
        fname = mol_name+'_hole_%04d.cube' %(iT)
        f.write('mol addfile {%s} type cube waitfor all\n' %fname)
    f.write('\n')

    f.write('mol delrep 0 top\n')
    f.write('mol representation CPK 1.700000 0.300000 20.000000 16.000000\n')
    f.write('mol color Name\n')
    f.write('mol material Opaque\n')
    f.write('mol addrep top\n\n')

    f.write('# store name of the isosurface representation (id=1) for later use\n')
    f.write('mol representation Isosurface %e 0.0 0.0 0.0\n' %(iso_elec))
    f.write('mol material Transparent\n')
    f.write('mol color ColorID %d\n' %(color_elec))
    f.write('mol addrep top\n\n')

    f.write('# store name of the isosurface representation (id=2) for later use\n')
    f.write('mol representation Isosurface %e 0.0 0.0 0.0\n' %(-iso_elec))
    f.write('mol material Transparent\n')
    f.write('mol color ColorID %d\n' %(color_elec))
    f.write('mol addrep top\n\n')

    f.write('# store name of the isosurface representation (id=3) for later use\n')
    f.write('mol representation Isosurface %e 0.0 0.0 0.0\n' %(iso_hole))
    f.write('mol material Transparent\n')
    f.write('mol color ColorID %d\n' %(color_hole))
    f.write('mol addrep top\n\n')

    f.write('# store name of the isosurface representation (id=4) for later use\n')
    f.write('mol representation Isosurface %e 0.0 0.0 0.0\n' %(-iso_hole))
    f.write('mol material Transparent\n')
    f.write('mol color ColorID %d\n' %(color_hole))
    f.write('mol addrep top\n\n')


    f.write('set updrep_plus [mol repname top 1]\n')
    f.write('set updrep_minus [mol repname top 2]\n\n')

    f.write('set updrep_plus_2 [mol repname top 3]\n')
    f.write('set updrep_minus_2 [mol repname top 4]\n\n')

    f.write('mol rename top {MOL_HF_DYN}\n')
    f.write('rotate z by -90\n')
    f.write('rotate x by -90\n')
    f.write('rotate x by 2\n')
    f.write('rotate y by 3\n')
    f.write('rotate x by 2\n\n')

    f.write('# use the volumetric data set for the isosurface corresponding to the frame.\n')
    f.write('# $updmol contains the id of the molecule and $updrep the (unique) name of \n')
    f.write('# the isosurface representation\n')
    
    f.write('proc update_iso {args} { \n')
    f.write('\n')
    f.write('    global updmol\n')
    f.write('    global updrep_plus\n')
    f.write('    global updrep_minus\n')
    f.write('    global updrep_plus_2\n')
    f.write('    global updrep_minus_2\n')
    f.write('\n')
    f.write('    # update representation but replace the data set \n')
    f.write('    # id with the current frame number.\n')
    f.write('    set frame [molinfo $updmol get frame]\n')
    f.write('    set numframe [molinfo $updmol get numframes]\n')
    f.write('    set frame2 [expr {$numframe/2+$frame}]\n')
    f.write('\n')
    f.write('    # change the color to the one we want\n')
    f.write('    mol color ColorID %d\n\n' %(color_elec))
    f.write('    # change the plus\n')
    f.write('    set repid_plus [mol repindex $updmol $updrep_plus]\n')
    f.write('    if {$repid_plus < 0} { return }\n')
    f.write('    lassign [molinfo $updmol get "{rep $repid_plus}"] rep_plus\n')
    f.write('    mol representation [lreplace $rep_plus 2 2 $frame]\n')
    f.write('    mol modrep $repid_plus $updmol\n')
    f.write('\n')
    f.write('    # change the minus\n')
    f.write('    set repid_minus [mol repindex $updmol $updrep_minus]\n')
    f.write('    if {$repid_minus < 0} { return }\n')
    f.write('    lassign [molinfo $updmol get "{rep $repid_minus}"] rep_minus\n')
    f.write('    mol representation [lreplace $rep_minus 2 2 $frame] \n')
    f.write('    mol modrep $repid_minus $updmol\n')
    f.write('\n')
    f.write('    # change the color to the one we want\n')
    f.write('    mol color ColorID %d\n\n' %(color_hole))
    f.write('    # change the plus\n')
    f.write('    set repid_plus [mol repindex $updmol $updrep_plus_2]\n')
    f.write('    if {$repid_plus < 0} { return }\n')
    f.write('    lassign [molinfo $updmol get "{rep $repid_plus}"] rep_plus\n')
    f.write('    mol representation [lreplace $rep_plus 2 2 $frame2]\n')
    f.write('    mol modrep $repid_plus $updmol\n')
    f.write('\n')
    f.write('    # change the minus\n')
    f.write('    set repid_minus [mol repindex $updmol $updrep_minus_2]\n')
    f.write('    if {$repid_minus < 0} { return }\n')
    f.write('    lassign [molinfo $updmol get "{rep $repid_minus}"] rep_minus\n')
    f.write('    mol representation [lreplace $rep_minus 2 2 $frame2] \n')
    f.write('    mol modrep $repid_minus $updmol\n')
    f.write('\n')
    f.write('\n')    
    f.write('}\n')
    f.write('\n')
    f.write('trace variable vmd_frame($updmol) w update_iso\n')
    f.write('animate goto 0\n')
    f.write('\n')
    f.close()

####################################################
#          Create the script
#            for blender mo
####################################################
def create_blender_script_mo(blname,xyz_min,nb_pts,spacing,pdbfile,bvoxfiles,path_to_files):

    # center/size of the 
    xyz_max = xyz_min+(nb_pts-1)*spacing
    
    center = bohr2ang*0.5*(xyz_max+xyz_min)/10
    size = bohr2ang*(xyz_max-xyz_min)/10/2

    f = open(blname,'w')
    f.write('import blmol\n')
    f.write('import bpy\n')
    f.write('import os\n\n')
    f.write("#go in the directories where the files are\n")
    f.write("#you may want to change that if the rendering\n")
    f.write("#and the calculations are not done on the same machine\n")
    f.write("os.chdir('%s')\n" %(path_to_files))
    f.write("\n")
    f.write("############################\n")
    f.write("# Tags to initialize\n")
    f.write("# or render the orbitals\n")
    f.write("############################\n")
    f.write("# keep _init_scene_ set to 1 and run the script\n")
    f.write("# Tweek the scene to your likings\n")
    f.write("# set _init_scene_=0 and _render_scene_=1 \n")
    f.write("# and re-run the script to render the MOs \n")
    f.write("############################\n")
    f.write('_init_scene_=1\n')
    f.write('_render_scene_=0\n')
    f.write("\n")
    f.write("\n")
    f.write("############################\n")
    f.write("# function to create the material\n")
    f.write("# we want for the orbitals\n")
    f.write("############################\n")
    f.write("def makeMaterial(name_mat,name_text,bvoxfile):\n")
    f.write("\n")
    f.write("    # create the material\n")
    f.write("    mat = bpy.data.materials.new(name_mat)\n")
    f.write("    mat.type = 'VOLUME'\n")
    f.write("    mat.volume.density = 0.0\n")
    f.write("    mat.volume.emission = 10.0\n")
    f.write("\n")
    f.write("    # create the texture\n")
    f.write("    text = bpy.data.textures.new(name_text,type='VOXEL_DATA')\n")
    f.write("    text.voxel_data.file_format = 'BLENDER_VOXEL'\n")
    f.write("    text.voxel_data.filepath = bvoxfile\n")
    f.write("    text.voxel_data.use_still_frame = True\n")
    f.write("    text.voxel_data.still_frame = 1\n")
    f.write("    text.voxel_data.intensity = 10\n")
    f.write("    \n")
    f.write("    # use a color ramp\n")
    f.write("    text.use_color_ramp = True\n")
    f.write("    text.color_ramp.elements.new(0.5)\n")
    f.write("    text.color_ramp.elements[0].color = (0.0,0.0,0.0,0.0)\n")
    f.write("    text.color_ramp.elements[1].color = (0.0,0.30,0.50,1.0)\n")
    f.write("    text.color_ramp.elements[2].color = (1.0,1.0,1.0,0.0)\n")
    f.write("    \n")
    f.write("    # add the texture to the amt\n")
    f.write("    mtex = mat.texture_slots.add()\n")
    f.write("    mtex.texture = text\n")
    f.write("    mtex.texture_coords = 'ORCO'\n")
    f.write("    mtex.use_map_density = True\n")
    f.write("    mtex.use_map_emission = True\n")
    f.write("    mtex.use_from_dupli = False\n")
    f.write("    mtex.use_map_to_bounds = False\n")
    f.write("    mtex.use_rgb_to_intensity = False\n")
    f.write("    \n")
    f.write("    # return the mat\n")
    f.write("    return mat\n")
    f.write("############################\n")
    f.write("\n")
    f.write("############################\n")
    f.write("# function to assigne a material\n")
    f.write("# to an oject\n")
    f.write("############################\n")
    f.write("def setMaterial(ob, mat):\n")
    f.write("    me = ob.data\n")
    f.write("    me.materials.append(mat)\n")
    f.write("############################\n")
    f.write("\n")
    f.write("\n")
    f.write("############################\n")
    f.write("# Function to initialize \n")
    f.write("# the Blender scene \n")
    f.write("############################\n")
    f.write("def init_scene():\n")
    f.write("\n")
    f.write("   # change the render setting\n")
    f.write("   bpy.context.scene.render.resolution_x = 2000\n")
    f.write("   bpy.context.scene.render.resolution_y = 2000\n")
    f.write("\n")
    f.write("   # change the horizon color\n")
    f.write("   bpy.context.scene.world.horizon_color = (0,0,0)\n")
    f.write("\n")
    f.write("   # change the camera position\n")
    f.write("   cam = bpy.data.objects['Camera']\n")
    f.write("   cam.location.x = %f\n" %center[0])
    f.write("   cam.location.y = %f\n" %center[1])
    f.write("   cam.location.z = %f\n" %(center[2]+2.))
    f.write("   cam.rotation_euler[0] = 0\n")
    f.write("   cam.rotation_euler[1] = 0\n")
    f.write("   cam.rotation_euler[2] = 0\n")
    f.write("\n")
    f.write("   #Create the molecule\n")
    f.write('   m = blmol.Molecule()\n')
    f.write("   m.read_pdb('%s')\n" %(pdbfile))
    f.write('   m.draw_bonds()\n')
    f.write("\n")
    f.write("   #Create the material\n")
    f.write("   momat = makeMaterial('momat','motext','%s')\n\n" %(bvoxfiles[0]))
    f.write("   # create the cube that we use to display the volume\n")    
    f.write('   bpy.ops.mesh.primitive_cube_add(location=(%f,%f,%f))\n' %(center[0],center[1],center[2]))
    f.write('   bpy.ops.transform.rotate(value=%f,axis=(0.0,1.,0.0))\n' %(np.pi/2))
    f.write('   bpy.ops.transform.resize(value=(%f,%f,%f))\n' %(size[0],size[1],size[2]))
    f.write("\n")
    f.write("   # Assign the material to the cube\n") 
    f.write('   setMaterial(bpy.context.object, momat)\n')
    f.write("############################\n")
    f.write("# Function to render all \n")
    f.write("# the bvoxfiles \n")
    f.write("############################\n")
    f.write("def render_files():\n")
    for iF in range(len(bvoxfiles)):
        image_name = bvoxfiles[iF][:-4]+'jpg'
        f.write("  bpy.data.textures['motext'].voxel_data.filepath = '%s' \n" %(bvoxfiles[iF]))
        f.write("  bpy.data.scenes['Scene'].render.filepath = '%s'\n" %image_name)
        f.write("  bpy.ops.render.render( write_still=True )\n\n")
    f.write("############################\n")
    f.write("\n")
    f.write("############################\n")
    f.write("# switch to init or render the scene\n")
    f.write("############################\n")
    f.write("if _init_scene_ == 1:\n")
    f.write("   init_scene()\n")
    f.write("elif _render_scene_==1:\n")
    f.write("   render_files()\n")
    f.write("############################\n")
    f.close()

####################################################
#          Create the script
#            for blender trajectory
####################################################
def create_blender_script_traj(fname,xyz_min,nb_pts,spacing,pdbfile,bvoxfiles_elec,bvoxfiles_hole,path_to_files):

    # center/size of the 
    xyz_max = xyz_min+(nb_pts-1)*spacing
    
    center = bohr2ang*0.5*(xyz_max+xyz_min)/10
    size = bohr2ang*(xyz_max-xyz_min)/10/2

    f = open(fname,'w')
    f.write('import blmol\n')
    f.write('import bpy\n')
    f.write('import os\n\n')
    f.write("#go in the directories where the files are\n")
    f.write("#you may want to change that if the rendering\n")
    f.write("#and the calculations are not done on the same machine\n")
    f.write("os.chdir('%s')\n" %(path_to_files))
    f.write("\n")
    f.write("############################\n")
    f.write("# Tags to initialize\n")
    f.write("# or render the orbitals\n")
    f.write("############################\n")
    f.write("# keep _init_scene_ set to 1 and run the script\n")
    f.write("# Tweek the scene to your likings\n")
    f.write("# set _init_scene_=0 and _render_scene_=1 \n")
    f.write("# and re-run the script to render the MOs \n")
    f.write("############################\n")
    f.write('_init_scene_=1\n')
    f.write('_render_scene_=0\n')
    f.write("\n")
    f.write("\n")
    f.write("############################\n")
    f.write("# function to create the material\n")
    f.write("# we want for the orbitals\n")
    f.write("############################\n")
    f.write("def makeMaterial(name_mat,name_text_elec,name_text_hole,bvoxfile_elec,bvoxfile_hole,color_elec,color_hole):\n")
    f.write("\n")
    f.write("    # create the material\n")
    f.write("    mat = bpy.data.materials.new(name_mat)\n")
    f.write("    mat.type = 'VOLUME'\n")
    f.write("    mat.volume.density = 0.0\n")
    f.write("    mat.volume.emission = 10.0\n")
    f.write("\n")
    f.write("    # create the texture for the electron\n")
    f.write("    text_elec = bpy.data.textures.new(name_text_elec,type='VOXEL_DATA')\n")
    f.write("    text_elec.voxel_data.file_format = 'BLENDER_VOXEL'\n")
    f.write("    text_elec.voxel_data.filepath = bvoxfile_elec\n")
    f.write("    text_elec.voxel_data.use_still_frame = True\n")
    f.write("    text_elec.voxel_data.still_frame = 1\n")
    f.write("    text_elec.voxel_data.intensity = 10\n")
    f.write("    \n")
    f.write("    # use a color ramp\n")
    f.write("    text_elec.use_color_ramp = True\n")
    f.write("    text_elec.color_ramp.elements.new(0.5)\n")
    f.write("    text_elec.color_ramp.elements[0].color = (0.0,0.0,0.0,0.0)\n")
    f.write("    text_elec.color_ramp.elements[1].color = color_elec\n")
    f.write("    text_elec.color_ramp.elements[2].color = (1.0,1.0,1.0,0.0)\n")
    f.write("    \n")
    f.write("    # create the texture for the hole\n")
    f.write("    text_hole = bpy.data.textures.new(name_text_hole,type='VOXEL_DATA')\n")
    f.write("    text_hole.voxel_data.file_format = 'BLENDER_VOXEL'\n")
    f.write("    text_hole.voxel_data.filepath = bvoxfile_hole\n")
    f.write("    text_hole.voxel_data.use_still_frame = True\n")
    f.write("    text_hole.voxel_data.still_frame = 1\n")
    f.write("    text_hole.voxel_data.intensity = 10\n")
    f.write("    \n")
    f.write("    # use a color ramp\n")
    f.write("    text_hole.use_color_ramp = True\n")
    f.write("    text_hole.color_ramp.elements.new(0.5)\n")
    f.write("    text_hole.color_ramp.elements[0].color = (0.0,0.0,0.0,0.0)\n")
    f.write("    text_hole.color_ramp.elements[1].color = color_hole\n")
    f.write("    text_hole.color_ramp.elements[2].color = (1.0,1.0,1.0,0.0)\n")
    f.write("    \n")
    f.write("    # add the electron texture to the mat\n")
    f.write("    mtex = mat.texture_slots.add()\n")
    f.write("    mtex.texture = text_elec\n")
    f.write("    mtex.texture_coords = 'ORCO'\n")
    f.write("    mtex.use_map_density = True\n")
    f.write("    mtex.use_map_emission = True\n")
    f.write("    mtex.use_from_dupli = False\n")
    f.write("    mtex.use_map_to_bounds = False\n")
    f.write("    mtex.use_rgb_to_intensity = False\n")
    f.write("    \n")
    f.write("    # add the hole texture to the mat\n")
    f.write("    mtex = mat.texture_slots.add()\n")
    f.write("    mtex.texture = text_hole\n")
    f.write("    mtex.texture_coords = 'ORCO'\n")
    f.write("    mtex.use_map_density = True\n")
    f.write("    mtex.use_map_emission = True\n")
    f.write("    mtex.use_from_dupli = False\n")
    f.write("    mtex.use_map_to_bounds = False\n")
    f.write("    mtex.use_rgb_to_intensity = False\n")
    f.write("    \n")
    f.write("    # return the mat\n")
    f.write("    return mat\n")
    f.write("############################\n")
    f.write("\n")
    f.write("############################\n")
    f.write("# function to assigne a material\n")
    f.write("# to an oject\n")
    f.write("############################\n")
    f.write("def setMaterial(ob, mat):\n")
    f.write("    me = ob.data\n")
    f.write("    me.materials.append(mat)\n")
    f.write("############################\n")
    f.write("\n")
    f.write("\n")
    f.write("############################\n")
    f.write("# Function to initialize \n")
    f.write("# the Blender scene \n")
    f.write("############################\n")
    f.write("def init_scene():\n")
    f.write("\n")
    f.write("   # change the render setting\n")
    f.write("   bpy.context.scene.render.resolution_x = 2000\n")
    f.write("   bpy.context.scene.render.resolution_y = 2000\n")
    f.write("\n")
    f.write("   # change the horizon color\n")
    f.write("   bpy.context.scene.world.horizon_color = (0,0,0)\n")
    f.write("\n")
    f.write("   # change the camera position\n")
    f.write("   cam = bpy.data.objects['Camera']\n")
    f.write("   cam.location.x = %f\n" %center[0])
    f.write("   cam.location.y = %f\n" %center[1])
    f.write("   cam.location.z = %f\n" %(center[2]+2.))
    f.write("   cam.rotation_euler[0] = 0\n")
    f.write("   cam.rotation_euler[1] = 0\n")
    f.write("   cam.rotation_euler[2] = 0\n")
    f.write("\n")
    f.write("   #Create the molecule\n")
    f.write('   m = blmol.Molecule()\n')
    f.write("   m.read_pdb('%s')\n" %(pdbfile))
    f.write('   m.draw_bonds()\n')
    f.write("\n")
    f.write("   #########################################\n")
    f.write("\n")
    f.write("   #Create the material for the electronic/hole density\n")
    f.write("   color_elec = (0.0,0.30,0.50,1.0)\n")
    f.write("   color_hole = (0.3,0.0,0.50,1.0)\n")
    f.write("   #Create the material for the hole density\n")
    f.write("   momat = makeMaterial('momat','motext_elec','motext_hole','%s','%s',color_elec,color_hole)\n" %(bvoxfiles_elec[0],bvoxfiles_hole[0]))
    f.write("\n")
    f.write("   # create the cube that we use to display the volume\n")    
    f.write('   bpy.ops.mesh.primitive_cube_add(location=(%f,%f,%f))\n' %(center[0],center[1],center[2]))
    f.write("   bpy.context.active_object.name = 'Cube_dens'\n")
    f.write('   bpy.ops.transform.rotate(value=%f,axis=(0.0,1.,0.0))\n' %(np.pi/2))
    f.write('   bpy.ops.transform.resize(value=(%f,%f,%f))\n' %(size[0],size[1],size[2]))
    f.write("\n")
    f.write("   # Assign the material to the cube\n") 
    f.write("   setMaterial(bpy.data.objects['Cube_dens'], momat)\n")
    f.write("\n")
    f.write("   #########################################\n")
    f.write("\n")
    f.write("\n")
    f.write("############################\n")
    f.write("# Function to render all \n")
    f.write("# the bvoxfiles \n")
    f.write("############################\n")
    f.write("def render_files():\n")
    f.write("\n")
    for iF in range(len(bvoxfiles_elec)):
        f.write("   # ==== Snap %d\n\n" %iF)
        image_name = 'elec_%04d.png' %iF
        f.write("   # render the electronic density\n")
        f.write("   bpy.data.materials['momat'].use_textures[0]=True\n")
        f.write("   bpy.data.materials['momat'].use_textures[1]=False\n")
        f.write("   bpy.data.textures['motext_elec'].voxel_data.filepath = '%s' \n" %(bvoxfiles_elec[iF]))
        f.write("   bpy.data.scenes['Scene'].render.filepath = '%s'\n" %image_name)
        f.write("   bpy.ops.render.render( write_still=True )\n\n")
        image_name = 'hole_%04d.png' %iF
        f.write("   # render the hole density\n")
        f.write("   bpy.data.materials['momat'].use_textures[0]=False\n")
        f.write("   bpy.data.materials['momat'].use_textures[1]=True\n")
        f.write("   bpy.data.textures['motext_hole'].voxel_data.filepath = '%s' \n" %(bvoxfiles_hole[iF]))
        f.write("   bpy.data.scenes['Scene'].render.filepath = '%s'\n" %image_name)
        f.write("   bpy.ops.render.render( write_still=True )\n\n")
    f.write("############################\n")
    f.write("\n")
    f.write("############################\n")
    f.write("# switch to init or render the scene\n")
    f.write("############################\n")
    f.write("if _init_scene_ == 1:\n")
    f.write("   init_scene()\n")
    f.write("elif _render_scene_==1:\n")
    f.write("   render_files()\n")
    f.write("############################\n")
    f.close()