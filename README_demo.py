READ ME

Set up to run the demo file:
    1. Import all support pakages: matplotlib, skimage, cv2, numpy, imutils
    2. Insure demo image file is in the same directory as the watershead code (uwatershead_ind_2color.py)
Run Segmentation 
    1. Run the watershead segmentation code (uwatershead_ind_2color.py) 
    2. Move segmented particles (all the generated png image file of individual rods) into a directory with the cntrl_file and lables code
        - Note: Depending on the image the scale bar or agrigated particles may have been captured. Do not take these into the new directory.
        Clustered particles will generate false catigories and images without regions (eg. only white from the border) defined in cntrl_file will cause the code to fail 
Run Lables
    1. Insure that the HSV color codes in the cntrl_file match the regions of intrest in the particle
    2. Run the lables code (Lables_v2.5.py), calling the cntrl_file from the command line 
    3. The code will generate lists of particles in the same catigory and the number in each catigory as well as an array iforming you of the regions and the order they were found in
       For example:"demo.jpg.14 as array (25.,102), array (94, 102)" means the particle represented in demo.jpg.14 is a rod with a blue region and a green region, which is a CdS-ZnS rod

Optimization for Files beyond the demo file:
    1. Optimzie the thresholding, itterations of opening and closing to optain cleanly segemented particles
            Look at the documentation for opencv as needed
    2. Optimize the cutoff (line 304 of Lables_v2.5.py) to remove noise but not real regions
    3. Change the cntrl_file to match the HSV regions you care about (see HSV color wheel on github) 
