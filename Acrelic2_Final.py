
# coding: utf-8

# In[21]:

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

print('Loading DICOM files...')
PathDicom = "C:\\Users\\L3IN\\Downloads\\Compressed\\2012.09.15 ACRELIC2 CT Scan Data from ACTREC\\09171420"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        lstFilesDCM.append(os.path.join(dirName,filename))
        
# Get ref file
RefDs = dicom.read_file(lstFilesDCM[0])

# Load dimensions based on the number of rows, columns, and slices (along the Z axis)
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

# Load spacing values (in mm)
ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

x = numpy.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
y = numpy.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
z = numpy.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])

# The array is sized based on 'ConstPixelDims'
Dicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

# loop through all the DICOM files
for filenameDCM in lstFilesDCM:
    # read the file
    ds = dicom.read_file(filenameDCM)
    # store the raw image data
    Dicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array
print('Processing slices...')   
RefImg = Dicom[:,:,0]
RefImg = RefImg > filters.threshold_otsu(RefImg)
RefImg = feature.canny(RefImg, sigma = 1)
RefImg = ndimage.binary_closing(RefImg)
RefImg = morphology.skeletonize(RefImg)
coords_corners = feature.corner_peaks(feature.corner_harris(RefImg), min_distance=7)

RefImg = Dicom[:,:,191]
RefImg = RefImg > filters.threshold_otsu(RefImg)
RefImg = feature.canny(RefImg, sigma = 2)
RefImg = ndimage.binary_closing(RefImg)
RefImg = morphology.skeletonize(RefImg)
coords_corners_left = feature.corner_peaks(feature.corner_harris(RefImg), min_distance=7)

RefImg = Dicom[:,:,100]
RefImg = RefImg > filters.threshold_otsu(RefImg)
RefImg = feature.canny(RefImg, sigma = 2)
RefImg = ndimage.binary_closing(RefImg)
RefImg = morphology.skeletonize(RefImg)
coords_corners_right = feature.corner_peaks(feature.corner_harris(RefImg), min_distance=7)

fiducials = []

for i in range(ConstPixelDims[2]):
    thirdCoord = i*ConstPixelSpacing[2]
    if Dicom[:,:,i].max() < 1000 or Dicom[:,:,i].max() > 2000:
        continue
    RefImg = Dicom[:,:,i]
    RefImg = RefImg > filters.threshold_otsu(RefImg)
    RefImg = feature.canny(RefImg, sigma = 1)
    RefImg = ndimage.binary_closing(RefImg)
    RefImg = morphology.skeletonize(RefImg)
    coords = feature.corner_peaks(feature.corner_harris(RefImg), min_distance=7)
    new_coords = []
    for i in range(len(coords)):
        for j in range(len(coords)):
            if j < i and numpy.sqrt(((coords[i][0] - coords[j][0])*ConstPixelSpacing[0])**2 + ((coords[i][1] - coords[j][1])*ConstPixelSpacing[1])**2) < 4.5:
                flag1 = 0
                flag2 = 0
                for k in range(len(coords_corners)):
                    if numpy.sqrt((coords[i][0] - coords_corners[k][0])**2 + (coords[i][1] - coords_corners[k][1])**2) < 5:
                        flag1 = 1
                    if numpy.sqrt((coords[j][0] - coords_corners[k][0])**2 + (coords[j][1] - coords_corners[k][1])**2) < 5:
                        flag2 = 1
                for k in range(len(coords_corners_left)):
                    if numpy.sqrt((coords[i][0] - coords_corners_left[k][0])**2 + (coords[i][1] - coords_corners_left[k][1])**2) < 27:
                        flag1 = 1
                    if numpy.sqrt((coords[j][0] - coords_corners_left[k][0])**2 + (coords[j][1] - coords_corners_left[k][1])**2) < 27:
                        flag2 = 1
                for k in range(len(coords_corners_right)):
                    if numpy.sqrt((coords[i][0] - coords_corners_right[k][0])**2 + (coords[i][1] - coords_corners_right[k][1])**2) < 27:
                        flag1 = 1
                    if numpy.sqrt((coords[j][0] - coords_corners_right[k][0])**2 + (coords[j][1] - coords_corners_right[k][1])**2) < 27:
                        flag2 = 1
                if flag1 == 0:
                    new_coords.append(coords[i])
                if flag2 == 0:
                    new_coords.append(coords[j])
    new_coords = numpy.array(new_coords)
    if len(new_coords) <= 1:
        continue
    new_coords = numpy.unique(new_coords, axis=0)
    if len(new_coords) <= 1:
        continue
    thresh = 30
    clusters = hcluster.fclusterdata(new_coords, thresh, criterion="distance")

        
    j = 1
    while j <= clusters.max():
        c1 = new_coords[clusters == j , 0].mean()
        c2 = new_coords[clusters == j , 1].mean()
        fiducials.append([c1,c2,thirdCoord])
        j += 1
        
fiducials = numpy.array(fiducials)

for i in range(len(coords_corners_left)):
    coords_corners_left[i][0] += 5
    
for i in range(len(coords_corners_right)):
    coords_corners_right[i][0] -= 10
    
print('Clustering and producing fiducial coordinates...')
db = DBSCAN(eps=9.8, min_samples=4).fit(fiducials)
clusters = db.labels_

new = []

for i in range(clusters.max()+1):
    c1 = fiducials[clusters == i , 0].mean()
    c2 = fiducials[clusters == i , 1].mean()
    c3 = fiducials[clusters == i , 2].mean()
    new.append([c1,c2,c3])
    
new = numpy.array(new)

final_fiducials = []

for i in range(len(new)):
    flag = 0
    for k in range(len(coords_corners_left)):
        if numpy.sqrt((new[i][0] - coords_corners_left[k][0])**2 + (new[i][1] - coords_corners_left[k][1])**2) < 29:
            flag = 1
    for k in range(len(coords_corners_right)):
        if numpy.sqrt((new[i][0] - coords_corners_right[k][0])**2 + (new[i][1] - coords_corners_right[k][1])**2) < 29:
            flag = 1
    if flag == 0:
        final_fiducials.append([new[i][1],new[i][0],new[i][2]])

final_fiducials = numpy.array(final_fiducials)

final_fiducials[:,1] = (ConstPixelDims[1] - final_fiducials[:,1])*ConstPixelSpacing[1]
final_fiducials[:,0] = final_fiducials[:,0]*ConstPixelSpacing[0]

print('\n\n-----------------Results----------------')
print("No. of fiducials : ", len(final_fiducials))
print(final_fiducials)

