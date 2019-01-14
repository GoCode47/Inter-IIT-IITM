
# coding: utf-8

# In[ ]:

import dicom
import os
import numpy
from matplotlib import pyplot, cm
from scipy import ndimage
from skimage import feature
from skimage import segmentation
from skimage import filters
from skimage import measure
from skimage import morphology
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
import scipy.cluster.hierarchy as hcluster
from mpl_toolkits.mplot3d import Axes3D

print('Loading DICOM files...')
PathDicom = "C:\\Users\\L3IN\\Downloads\\Compressed\\2012.09.15 ACRELIC1 CT Scan Data from ACTREC\\09171700"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    fileList.remove('14484183')
    fileList.remove('14483699')
    for filename in fileList:
        lstFilesDCM.append(os.path.join(dirName,filename))

# Get ref file
RefDs = dicom.read_file(lstFilesDCM[0])

# Load dimensions based on the number of rows, columns, and slices (along the Z axis)
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

# Load spacing values (in mm)
ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

Plane = 0
Sagittal = 0
Coronal = 0
for filename in lstFilesDCM:
    ds = dicom.read_file(filename)
    if ds.SeriesDescription == 'AXIAL 0.5':
        Plane += 1
    elif ds.SeriesDescription == 'SAG 0.3':
        Sagittal += 1
    else:
        Coronal += 1
        
x = numpy.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
y = numpy.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
z = numpy.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])

# The array is sized based on 'ConstPixelDims'
PlaneDicom = numpy.zeros((ConstPixelDims[0], ConstPixelDims[1], Plane), dtype=RefDs.pixel_array.dtype)
SagittalDicom = numpy.zeros((ConstPixelDims[0], ConstPixelDims[1], Sagittal), dtype=RefDs.pixel_array.dtype)
CoronalDicom = numpy.zeros((ConstPixelDims[0], ConstPixelDims[1], Coronal), dtype=RefDs.pixel_array.dtype)

# loop through all the DICOM files
i = 0
j = 0 
k = 0
for filenameDCM in lstFilesDCM:
    # read the file
    ds = dicom.read_file(filenameDCM)
    # store the raw image data
    if ds.SeriesDescription == 'AXIAL 0.5':
        PlaneDicom[:, :, i] = ds.pixel_array
        i += 1
    elif ds.SeriesDescription == 'SAG 0.3':
        SagittalDicom[:, :, j] = ds.pixel_array
        j += 1
    else:
        CoronalDicom[:, :, k] = ds.pixel_array
        k += 1

print('Processing Axial slices...')
RefImg_plane = PlaneDicom[:,:,0]
RefImg_plane = RefImg_plane > filters.threshold_otsu(RefImg_plane)
RefImg_plane = feature.canny(RefImg_plane, sigma = 1)
RefImg_plane = ndimage.binary_closing(RefImg_plane)
RefImg_plane = morphology.skeletonize(RefImg_plane)
coords_corners = feature.corner_peaks(feature.corner_harris(RefImg_plane), min_distance=7)

fiducials = []

for i in range(Plane):
    temp = PlaneDicom[:,:,i]
    temp = temp > filters.threshold_otsu(temp)
    temp = feature.canny(temp, sigma = 1)
    temp = ndimage.binary_closing(temp)
    temp = morphology.skeletonize(temp)
    coords = feature.corner_peaks(feature.corner_harris(temp), min_distance=7)
    
    new_coords = []
    
    for i in range(len(coords)):
        for j in range(len(coords)):
            if j < i and numpy.sqrt(((coords[i][0] - coords[j][0])*ConstPixelSpacing[0])**2 + ((coords[i][1] - coords[j][1])*ConstPixelSpacing[1])**2) < 12.5:
                flag1 = 0
                flag2 = 0
                for k in range(len(coords_corners)):
                    if numpy.sqrt((coords[i][0] - coords_corners[k][0])**2 + (coords[i][1] - coords_corners[k][1])**2) < 5:
                        flag1 = 1
                    if numpy.sqrt((coords[j][0] - coords_corners[k][0])**2 + (coords[j][1] - coords_corners[k][1])**2) < 5:
                        flag2 = 1
                if flag1 == 0:
                    new_coords.append(coords[i])
                if flag2 == 0:
                    new_coords.append(coords[j])
                    
    new_coords = numpy.array(new_coords)
    
    if len(new_coords) <= 1:
        continue
        
    new_coords = numpy.unique(new_coords, axis=0)
    
    thresh = 40
    clusters = hcluster.fclusterdata(new_coords, thresh, criterion="distance")
    
    j = 1
    while j <= clusters.max():
        c1 = new_coords[clusters == j , 0].mean()
        c2 = new_coords[clusters == j , 1].mean()
        fiducials.append([c1,c2])
        j += 1

fiducials = numpy.array(fiducials)
print('Clustering and producing Axial 2D coordinates...')
db = DBSCAN(eps=7, min_samples=2).fit(fiducials)
clusters = db.labels_

fiducial_plane = []

for i in range(clusters.max()+1):
    c1 = fiducials[clusters == i , 0].mean()
    c2 = fiducials[clusters == i , 1].mean()
    fiducial_plane.append([c1,c2])
    
fiducial_plane = numpy.array(fiducial_plane)

fiducial_plane[:,0] = (ConstPixelDims[0] - fiducial_plane[:,0])*ConstPixelSpacing[0]
fiducial_plane[:,1] = fiducial_plane[:,1]*ConstPixelSpacing[1]

print('Processing Coronal slices...')
RefImg_coronal = CoronalDicom[:,:,0]
RefImg_coronal = RefImg_coronal > filters.threshold_otsu(RefImg_coronal)
RefImg_coronal = feature.canny(RefImg_coronal, sigma = 1)
RefImg_coronal = ndimage.binary_closing(RefImg_coronal)
RefImg_coronal = morphology.skeletonize(RefImg_coronal)
coords_corners = feature.corner_peaks(feature.corner_harris(RefImg_coronal), min_distance=7)

fiducials_2 = []

for i in range(Coronal):
    
    if CoronalDicom[:,:,i].max() < 950:
        continue
        
    temp = CoronalDicom[:,:,i]
    temp = temp > filters.threshold_otsu(temp)
    temp = feature.canny(temp, sigma = 1)
    temp = ndimage.binary_closing(temp)
    temp = morphology.skeletonize(temp)
    coords = feature.corner_peaks(feature.corner_harris(temp), min_distance=7)
    new_coords = []
    for i in range(len(coords)):
        for j in range(len(coords)):
            if j < i and numpy.sqrt(((coords[i][0] - coords[j][0])*ConstPixelSpacing[0])**2 + ((coords[i][1] - coords[j][1])*ConstPixelSpacing[1])**2) < 12.5:
                flag1 = 0
                flag2 = 0
                for k in range(len(coords_corners)):
                    if numpy.sqrt((coords[i][0] - coords_corners[k][0])**2 + (coords[i][1] - coords_corners[k][1])**2) < 5:
                        flag1 = 1
                    if numpy.sqrt((coords[j][0] - coords_corners[k][0])**2 + (coords[j][1] - coords_corners[k][1])**2) < 5:
                        flag2 = 1
                if flag1 == 0:
                    new_coords.append(coords[i])
                if flag2 == 0:
                    new_coords.append(coords[j])
                    
    new_coords = numpy.array(new_coords)
    
    if len(new_coords) <= 1:
        continue
        
    new_coords = numpy.unique(new_coords, axis=0)
    
    thresh = 40
    clusters = hcluster.fclusterdata(new_coords, thresh, criterion="distance")
    
    j = 1
    while j <= clusters.max():
        c1 = new_coords[clusters == j , 0].mean()
        c2 = new_coords[clusters == j , 1].mean()
        fiducials_2.append([c1,c2])
        j += 1

fiducials_2 = numpy.array(fiducials_2)

print('Clustering and producing Coronal 2D coordinates...')
db = DBSCAN(eps=10, min_samples=4).fit(fiducials_2)
clusters = db.labels_

fiducial_coronal = []

for i in range(clusters.max()+1):
    c1 = fiducials_2[clusters == i , 0].mean()
    c2 = fiducials_2[clusters == i , 1].mean()
    fiducial_coronal.append([c1,c2])

fiducial_coronal = numpy.array(fiducial_coronal)

fiducial_coronal[:,0] = (ConstPixelDims[0] - fiducial_coronal[:,0])*ConstPixelSpacing[0]
fiducial_coronal[:,1] = fiducial_coronal[:,1]*ConstPixelSpacing[1]

print('Processing Sagittal slices...')
RefImg_sagittal = SagittalDicom[:,:,0]
RefImg_sagittal = RefImg_sagittal > filters.threshold_otsu(RefImg_sagittal)
RefImg_sagittal = feature.canny(RefImg_sagittal, sigma = 1)
RefImg_sagittal = ndimage.binary_closing(RefImg_sagittal)
RefImg_sagittal = morphology.skeletonize(RefImg_sagittal)
coords_corners = feature.corner_peaks(feature.corner_harris(RefImg_sagittal), min_distance=7)

fiducials_3 = []

for i in range(Sagittal):
    if SagittalDicom[:,:,i].max() < 500:
        continue
    temp = SagittalDicom[:,:,i]
    temp = temp > filters.threshold_otsu(temp)
    temp = feature.canny(temp, sigma = 1)
    temp = ndimage.binary_closing(temp)
    temp = morphology.skeletonize(temp)
    coords = feature.corner_peaks(feature.corner_harris(temp), min_distance=7)
    new_coords = []
    for i in range(len(coords)):
        for j in range(len(coords)):
            if j < i and numpy.sqrt(((coords[i][0] - coords[j][0])*ConstPixelSpacing[0])**2 + ((coords[i][1] - coords[j][1])*ConstPixelSpacing[1])**2) < 12.5:
                flag1 = 0
                flag2 = 0
                for k in range(len(coords_corners)):
                    if numpy.sqrt((coords[i][0] - coords_corners[k][0])**2 + (coords[i][1] - coords_corners[k][1])**2) < 5:
                        flag1 = 1
                    if numpy.sqrt((coords[j][0] - coords_corners[k][0])**2 + (coords[j][1] - coords_corners[k][1])**2) < 5:
                        flag2 = 1
                if flag1 == 0:
                    new_coords.append(coords[i])
                if flag2 == 0:
                    new_coords.append(coords[j])
                    
    new_coords = numpy.array(new_coords)
    
    if len(new_coords) <= 1:
        continue
        
    new_coords = numpy.unique(new_coords, axis=0)
    
    thresh = 40
    clusters = hcluster.fclusterdata(new_coords, thresh, criterion="distance")
    
    j = 1
    while j <= clusters.max():
        c1 = new_coords[clusters == j , 0].mean()
        c2 = new_coords[clusters == j , 1].mean()
        fiducials_3.append([c1,c2])
        j += 1

fiducials_3 = numpy.array(fiducials_3)
print('Clustering and producing Sagittal 2D coordinates...')
db = DBSCAN(eps=10, min_samples=4).fit(fiducials_3)
clusters = db.labels_

fiducial_sagittal = []

for i in range(0, clusters.max()+1):
    c1 = fiducials_3[clusters == i , 0].mean()
    c2 = fiducials_3[clusters == i , 1].mean()
    fiducial_sagittal.append([c1,c2])
    
fiducial_sagittal = numpy.array(fiducial_sagittal)

fiducial_sagittal[:,0] = (ConstPixelDims[0] - fiducial_sagittal[:,0])*ConstPixelSpacing[0]
fiducial_sagittal[:,1] = ConstPixelSpacing[1]*fiducial_sagittal[:,1]

print('Producing 3D fiducial coordinates...')
err = 9
FiducialCoords = []

for i in range(len(fiducial_plane)):
    for j in range(len(fiducial_coronal)):
        for k in range(len(fiducial_sagittal)):
            if numpy.absolute(fiducial_plane[i][0] - fiducial_sagittal[k][0]) < err and numpy.absolute(fiducial_plane[i][1] - fiducial_coronal[j][1]) < err and numpy.absolute(fiducial_coronal[j][0] - fiducial_sagittal[k][1]) < err:
                FiducialCoords.append([(fiducial_plane[i][1] + fiducial_coronal[j][1])/2.0 , (fiducial_plane[i][0] + fiducial_sagittal[k][0])/2.0 , (fiducial_coronal[j][0] + fiducial_sagittal[k][1])/2.0])
    
            
FiducialCoords = numpy.array(FiducialCoords)
print('\n\n-------------------------Results------------------------')
print("No. of fiducials : ", len(FiducialCoords))
print(FiducialCoords)


# In[ ]:



