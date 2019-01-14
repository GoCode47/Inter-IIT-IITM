
# coding: utf-8

# In[27]:

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

print('Loading DICOM images...')
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

for filename in lstFilesDCM:
    ds = dicom.read_file(filename)
    if ds.SeriesDescription == 'AXIAL 0.5':
        Plane += 1

x = numpy.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
y = numpy.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
z = numpy.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])

# The array is sized based on 'ConstPixelDims'
PlaneDicom = numpy.zeros((ConstPixelDims[0], ConstPixelDims[1], Plane), dtype=RefDs.pixel_array.dtype)

# loop through all the DICOM files
i = 0

for filenameDCM in lstFilesDCM:
    # read the file
    ds = dicom.read_file(filenameDCM)
    # store the raw image data
    if ds.SeriesDescription == 'AXIAL 0.5':
        PlaneDicom[:, :, i] = ds.pixel_array
        i += 1

print('Processing slices...')
RefImg_plane = PlaneDicom[:,:,0]
RefImg_plane = RefImg_plane > filters.threshold_otsu(RefImg_plane)
RefImg_plane = feature.canny(RefImg_plane, sigma = 1)
RefImg_plane = ndimage.binary_closing(RefImg_plane)
RefImg_plane = morphology.skeletonize(RefImg_plane)
coords_corners = feature.corner_peaks(feature.corner_harris(RefImg_plane), min_distance=7)

fiducials = []

for i in range(Plane):
    thirdcoord = (i*ConstPixelDims[0]*ConstPixelSpacing[0])/Plane
    RefImg = PlaneDicom[:,:,i]
    RefImg = RefImg > filters.threshold_otsu(RefImg)
    RefImg = feature.canny(RefImg, sigma = 1)
    RefImg = ndimage.binary_closing(RefImg)
    RefImg = morphology.skeletonize(RefImg)
    coords = feature.corner_peaks(feature.corner_harris(RefImg), min_distance=7)
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
        fiducials.append([c1*ConstPixelSpacing[0],c2*ConstPixelSpacing[1],thirdcoord])
        j += 1

fiducials = numpy.array(fiducials)

print('Clustering and producing fiducial coordinates...')
db = DBSCAN(eps=5, min_samples=6).fit(fiducials)
clusters = db.labels_

final_fiducials = []

for i in range(clusters.max()+1):
    c1 = fiducials[clusters == i , 0].mean()
    c2 = fiducials[clusters == i , 1].mean()
    c3 = fiducials[clusters == i , 2].mean()
    final_fiducials.append([c2,c1,c3])
    
final_fiducials = numpy.array(final_fiducials)

final_fiducials[:,1] = ConstPixelDims[0]*ConstPixelSpacing[0] - final_fiducials[:,1]

print('\n\n-----------------Results-----------------')
print("No. of fiducials : ", len(final_fiducials))
print(final_fiducials)

