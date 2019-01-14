
# coding: utf-8



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
PathDicom = "C:\\Users\\L3IN\\Downloads\\2016.06.27 PVC Skull Model\\2016.06.27 PVC Skull Model\\Sequential Scan\\DICOM\\PA1\\ST1\\SE2"
lstFilesDCM = []  
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        lstFilesDCM.append(os.path.join(dirName,filename))
        
RefDs = dicom.read_file(lstFilesDCM[0])

ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

x = numpy.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
y = numpy.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
z = numpy.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])

PlaneDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

# loop through all the DICOM files
for filenameDCM in lstFilesDCM:
    # read the file
    ds = dicom.read_file(filenameDCM)
    # store the raw image data
    PlaneDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array
    

print('Processing slices...')    
for k in range(ConstPixelDims[2]):
    for i in range(ConstPixelDims[0]):
        for j in range(ConstPixelDims[1]):
            if PlaneDicom[i,j,k] > 200 and PlaneDicom[i,j,k] < 15000:
                PlaneDicom[i,j,k] += 15000
                
fiducials = []

for i in range(ConstPixelDims[2]):
    thirdcoord = i * ConstPixelSpacing[2]
    PlaneDicom[:,:,i] = feature.canny(PlaneDicom[:,:,i],sigma=2)
    PlaneDicom[:,:,i] = ndimage.binary_closing(PlaneDicom[:,:,i])
    PlaneDicom[:,:,i] = morphology.skeletonize(PlaneDicom[:,:,i])
    coords = feature.corner_peaks(feature.corner_harris(PlaneDicom[:,:,i]), min_distance=7)
    new_coords = []
    for i in range(len(coords)):
        for j in range(len(coords)):
            if j < i and numpy.sqrt(((coords[i][0] - coords[j][0])*ConstPixelSpacing[0])**2 + ((coords[i][1] - coords[j][1])*ConstPixelSpacing[1])**2) < 7:
                new_coords.append(coords[i])
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

print('Clustering and producing Fiducial coordinates...')
db = DBSCAN(eps=4.5, min_samples=3).fit(fiducials)
clusters = db.labels_

final_fiducials = []

for i in range(0, clusters.max()+1):
    c1 = fiducials[clusters == i , 0].mean()
    c2 = fiducials[clusters == i , 1].mean()
    c3 = fiducials[clusters == i , 2].mean()
    final_fiducials.append([c2 ,c1, c3])

final_fiducials = numpy.array(final_fiducials)
    
final_fiducials[:,1] = ConstPixelDims[0]*ConstPixelSpacing[0] - final_fiducials[:,1]


print('\n\n------------Results----------------')
print("No. of fiducials : ", len(final_fiducials))
print(final_fiducials)


