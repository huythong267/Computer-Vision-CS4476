import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter

def get_interest_points(image, feature_width, alpha = 0.1, top= 1500):
    """
    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful in this function in order to (a) suppress boundary interest
    points (where a feature wouldn't fit entirely in the image, anyway)
    or (b) scale the image filters being used. Or you can ignore it.

    By default you do not need to make scale and orientation invariant
    local features.

    The lecture slides and textbook are a bit vague on how to do the
    non-maximum suppression once you've thresholded the cornerness score.
    You are free to experiment. For example, you could compute connected
    components and take the maximum value within each component.
    Alternatively, you could run a max() operator on each sliding window. You
    could use this to ensure that every interest point is at a local maximum
    of cornerness.

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   feature_width: integer representing the local feature width in pixels.

    Returns:
    -   x: A numpy array of shape (N,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N,) containing y-coordinates of interest points
    -   confidences (optional): numpy nd-array of dim (N,) containing the strength
            of each interest point
    -   scales (optional): A numpy array of shape (N,) containing the scale at each
            interest point
    -   orientations (optional): A numpy array of shape (N,) containing the orientation
            at each interest point
    """
    confidences, scales, orientations = None, None, None
    #############################################################################
    # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                                                      #
    #############################################################################
    
    # Define Gaussian Filter and alpha factor for cornerness R function
    gaussian =cv2.getGaussianKernel(ksize=5, sigma=5)
    gaussian = gaussian.dot(gaussian.T)

    #First calculate the derivative Ix, Iy with Sober operator
    Ix = cv2.Sobel(image, cv2.CV_32F,1,0,ksize=5)  # x
    Iy = cv2.Sobel(image, cv2.CV_32F,0,1,ksize=5)  # y
    
    #Second, calculate the second order derivative with filter
    Ix2, Ixy, Iy2 = Ix**2, Ix*Iy, Iy**2
    gIx2=cv2.filter2D(Ix2,-1,gaussian)
    gIxy=cv2.filter2D(Ixy,-1,gaussian)
    gIy2=cv2.filter2D(Iy2,-1,gaussian)
    
    # Calculate the R-function
    R = gIx2*gIy2 - gIxy**2 - alpha*(gIx2+gIy2)**2
    
    # zero-padding and non-maximal supression
    R = R*(R>0.001*R.max())
    R[:feature_width//2,:] = 0
    R[-feature_width//2:,:] = 0
    R[:,:feature_width//2] = 0
    R[:,-feature_width//2:] = 0
    
    R = R*(R ==  maximum_filter(R,(11,11)))
    
    # Sorted the indexes by R-value
    
    indx,indy = np.unravel_index(np.argsort(-R, axis=None), R.shape)
    max_len = np.sum((R>0)) #length of non-zero elements in R
    indx, indy = indx[:max_len], indy[:max_len]

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    #############################################################################
    # TODO: YOUR ADAPTIVE NON-MAXIMAL SUPPRESSION CODE HERE                     #
    # While most feature detectors simply look for local maxima in              #
    # the interest function, this can lead to an uneven distribution            #
    # of feature points across the image, e.g., points will be denser           #
    # in regions of higher contrast. To mitigate this problem, Brown,           #
    # Szeliski, and Winder (2005) only detect features that are both            #
    # local maxima and whose response value is significantly (10%)              #
    # greater than that of all of its neighbors within a radius r. The          #
    # goal is to retain only those points that are a maximum in a               #
    # neighborhood of radius r pixels. One way to do so is to sort all          #
    # points by the response strength, from large to small response.            #
    # The first entry in the list is the global maximum, which is not           #
    # suppressed at any radius. Then, we can iterate through the list           #
    # and compute the distance to each interest point ahead of it in            #
    # the list (these are pixels with even greater response strength).          #
    # The minimum of distances to a keypoint's stronger neighbors               #
    # (multiplying these neighbors by >=1.1 to add robustness) is the           #
    # radius within which the current point is a local maximum. We              #
    # call this the suppression radius of this interest point, and we           #
    # save these suppression radii. Finally, we sort the suppression            #
    # radii from large to small, and return the n keypoints                     #
    # associated with the top n suppression radii, in this sorted               #
    # orderself. Feel free to experiment with n, we used n=1500.                #
    #                                                                           #
    # See:                                                                      #
    # https://www.microsoft.com/en-us/research/wp-content/uploads/2005/06/cvpr05.pdf
    # or                                                                        #
    # https://www.cs.ucsb.edu/~holl/pubs/Gauglitz-2011-ICIP.pdf                 #
    #############################################################################

    def adaptive_non_maximal_supression(indx, indy, R, top= 1500, max_len = 3000):
        '''
            Input:
                indx, indy: coordinates of interest points
                R: corner function
            Output:
                sorted(indx, indy) by the criterion of adaptive none-maximal suppression
        '''
        indx, indy = indx[:max_len], indy[:max_len]
        X = np.c_[indx, indy]   #(max_len, 2)
        
        # Find out the distance_0p9 as the minimum distance from one point (x,y) 
        # to the next point (x_new,y_new) with R(x_new,y_new)>0.9*R(x,y)
        distance_0p9 = []

        for i, (x, y) in enumerate(zip(indx, indy)):
            corner_0p9 = 0.9*R[x,y]  #thereshold value = 0.9*R(x,y)
            
            #calculate the distance from to all other points (x_new,y_new) to this point (x,y) (or X[i]) 
            distance_i = np.linalg.norm(X- X[i], axis = 1)  # (max_len, 2) - (2,) = (max_len, 2) --> norm, axis = 0 --> (max_len,)
            
            # Filter out the points (x_new,y_new) where R(x_new,y_new) > corner_0p9
            distance_i_with_significant_R = np.sort(distance_i[R[indx, indy]>=corner_0p9])
            
            # We look for the closest point with R(x_new,y_new)> 0.9*R(x,y) but not the point (x,y)
            min_distance = distance_i_with_significant_R[1] if len(distance_i_with_significant_R)>1 else float('inf')

            distance_0p9.append([x,y,min_distance])
         

        #Sort this distance_0p9 from the min_distance
        distance_0p9.sort(key = lambda x: x[2], reverse = True)
        
        # Return the (x,y) coordinates 
        distance_0p9 = np.array(distance_0p9[:top])
        return distance_0p9[:,0], distance_0p9[:,1]
    
    # Reverse y, x for the image coordinates
    y, x = adaptive_non_maximal_supression(indx, indy, R, top = top)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return x.astype(int),y.astype(int), confidences, scales, orientations
