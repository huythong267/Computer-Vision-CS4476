import numpy as np
import cv2


def get_features(image, x, y, feature_width, scales=None):
    """
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    -   scales: Python list or tuple if you want to detect and describe features
            at multiple scales

    You may also detect and describe features at particular orientations.

    Returns:
    -   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    assert image.ndim == 2, 'Image must be grayscale'
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    # If you choose to implement rotation invariance, enabling it should not    #
    # decrease your matching accuracy.                                          #
    #############################################################################
    
    # Swap the coordinate to work with nupy array
    x, y = y, x
    
    size = int(feature_width//4)

    # Calculate the Ix, Iy gradient vector
    Ix = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=5)
    Iy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=5)
    
    # Calculate the descriptors
    descriptors = np.zeros((x.shape[0], 128))
    
    for i,(kx, ky)  in enumerate(zip(x,y)):
        descriptors[i] = single_128_feature_vector(Ix, Iy, int(kx), int(ky), size)
    
    return descriptors

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

def single_128_feature_vector(Ix, Iy, kx, ky, size):
    descriptor = []
    for ix in range(-2,2):
        for iy in range(-2,2):
            locx , locy =  kx + ix*size, ky + iy*size
            histogram8 = histogram8bins(Ix, Iy, int(locx) , int(locy), size)
            descriptor.append(histogram8)

    descriptor = np.array(descriptor).flatten()

    #Normalize the descriptor
    descriptor = descriptor / np.linalg.norm(descriptor) 
    descriptor = np.clip(descriptor, 0, 0.2)
    descriptor = descriptor / np.linalg.norm(descriptor)

    return descriptor


def histogram8bins(Ix, Iy, indx,indy, size):
    '''
        Given a window of image (square of: size x size)
        Return the total value of 8 bin histogram [0,45,90,135,180,225,270,325]
    '''
    histogram8 = np.zeros(8)
    for x in range(size):
        for y in range(size):
            histogram8 = histogram8 + pixel2histogram(Ix[indx+x, indy+y], Iy[indx+x, indy+y])

    return list(histogram8)

def pixel2histogram(ix, iy):
    '''
        Given a gradient of a pixel in Cart form (ix, iy), convert to 8 bin histogram [0,45,90,135,180,225,270,325]
        Input:
            ix, iy: gradient at a given pixel
        Return:
            Histogram of this new pixel: array same length as [0,45,90,135,180,225,270,325]
    '''
    # Convert to polar
    mag = np.sqrt(ix**2 + iy**2)
    phase = np.arctan2(iy,ix)*180/np.pi

    phase = phase + 360 if phase<0 else phase  # convert from -180 --> 180 to 0 to 360
    left =  int(phase//45)  
    right = int(phase//45 + 1)  # if phase>325, right = 360

    pixel_histogram = np.zeros(9)  # padding 360 deg at the end to transform to [...,225,270,325,360]
    # Separate the vector into 2 nearby angles
    pixel_histogram[right] =mag*(phase-45*left)/45
    pixel_histogram[left] =mag*(45*right-phase)/45
    pixel_histogram[0] += pixel_histogram[8]  # If the right = 8 or phase = 360, then it is equivalent to add to 0

    return pixel_histogram[:8]
