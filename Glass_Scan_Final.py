
# coding: utf-8

# In[33]:

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

print('Loading DICOM files')
PathDicom = "C:\\Users\\L3IN\\Downloads\\2016.05.26 Glass Scan 1 mm\\2016.05.26 Glass Scan 1 mm\\Glass Scan Axial 1.25 mm\\DICOM\\PA1\\ST1\\SE2"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        lstFilesDCM.append(os.path.join(dirName,filename))
        
# Get ref file
RefDs = dicom.read_file(lstFilesDCM[1])

# Load dimensions based on the number of rows, columns, and slices (along the Z axis)
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

# Load spacing values (in mm)
ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

x = numpy.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
y = numpy.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
z = numpy.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])

# The array is sized based on 'ConstPixelDims'
PlaneDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

# loop through all the DICOM files
for filenameDCM in lstFilesDCM:
    # read the file
    ds = dicom.read_file(filenameDCM)
    # store the raw image data
    PlaneDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array

print('Processing Slices')
fiducials = []

for i in range(ConstPixelDims[2]):
    thirdcoords = i*ConstPixelSpacing[2]*(ConstPixelDims[0]*ConstPixelSpacing[0])/(ConstPixelSpacing[2]*ConstPixelDims[2])
    img = PlaneDicom[:,:,i] > 10000
    img = ndimage.binary_erosion(img)
    coords = feature.corner_peaks(feature.corner_harris(img), min_distance=7)
    coords = numpy.array(coords)
    if coords.size == 0:
        continue
    elif len(coords) == 1:
        fiducials.append([coords[0][0]*ConstPixelSpacing[0], coords[0][1]*ConstPixelSpacing[0], thirdcoords])
        continue
    else:
        thresh = 40
        clusters = hcluster.fclusterdata(coords, thresh, criterion="distance")
    
    j = 1
    while j <= clusters.max():
        c1 = coords[clusters == j , 0].mean()
        c2 = coords[clusters == j , 1].mean()
        fiducials.append([c1*ConstPixelSpacing[0],c2*ConstPixelSpacing[1],thirdcoords])
        j += 1
        
fiducials = numpy.array(fiducials)
print('Clustering and producing fiducial coordinates...')
thresh = 3
clusters = hcluster.fclusterdata(fiducials, thresh, criterion="distance")

final_fiducials = []

for i in range(clusters.min(), clusters.max()+1):
    c1 = fiducials[clusters == i , 0].mean()
    c2 = fiducials[clusters == i , 1].mean()
    c3 = fiducials[clusters == i , 2].mean()
    final_fiducials.append([c2,c1,c3])
    
final_fiducials = numpy.array(final_fiducials)

final_fiducials[:,1] = ConstPixelDims[0]*ConstPixelSpacing[0] - final_fiducials[:,1]
print('\n\n---------------Results----------------')
print("No. of fiducials : ", len(final_fiducials))
print(final_fiducials)

