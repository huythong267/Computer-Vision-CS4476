import numpy as np
np.random.seed(2019)

def calculate_projection_matrix(points_2d, points_3d):
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points:

                                                      [ M11      [ u1
                                                        M12        v1
                                                        M13        .
                                                        M14        .
    [ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1        M21        .
      0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1        M22        .
      .  .  .  . .  .  .  .    .     .      .       *   M23   =    .
      Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn        M24        .
      0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]      M31        .
                                                        M32        un
                                                        M33 ]      vn ]

    Then you can solve this using least squares with np.linalg.lstsq() or SVD.
    Notice you obtain 2 equations for each corresponding 2D and 3D point
    pair. To solve this, you need at least 6 point pairs.

    Args:
    -   points_2d: A numpy array of shape (N, 2)
    -   points_2d: A numpy array of shape (N, 3)

    Returns:
    -   M: A numpy array of shape (3, 4) representing the projection matrix
    """

    # Placeholder M matrix. It leads to a high residual. Your total residual
    # should be less than 1.
    M = np.asarray([[0.1768, 0.7018, 0.7948, 0.4613],
                    [0.6750, 0.3152, 0.1136, 0.0480],
                    [0.1020, 0.1725, 0.7244, 0.9932]])

    ###########################################################################
    # TODO: YOUR PROJECTION MATRIX CALCULATION CODE HERE
    ###########################################################################

    assert len(points_2d) == len(points_3d)
    assert len(points_2d) >5
    
    # Initialize the data matrix
    N = len(points_2d)
    A = np.zeros([N*2,11])
    b = np.zeros([N*2,1])
    
    # Write down all the value inside the matrix
    for ii in range(N):
        X1, Y1, Z1 = points_3d[ii]
        u1, v1 = points_2d[ii]
        
        A[ii*2] = np.array([X1, Y1, Z1, 1, 0,  0,  0, 0, -u1*X1, -u1*Y1, -u1*Z1])
        A[ii*2+1] = np.array([0,  0,  0,  0, X1, Y1, Z1, 1, -v1*X1, -v1*Y1, -v1*Z1])
        b[ii*2] = u1
        b[ii*2+1] = v1
    
    # Solve for M matrix using the least square
    M, residuals, rank, s = np.linalg.lstsq(A,b,rcond=None)
    
    M = np.vstack([M,1.0]).reshape(3,4)

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return M

def calculate_camera_center(M):
    """
    Returns the camera center matrix for a given projection matrix.

    The center of the camera C can be found by:

        C = -Q^(-1)m4

    where your project matrix M = (Q | m4).

    Args:
    -   M: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """

    # Placeholder camera center. In the visualization, you will see this camera
    # location is clearly incorrect, placing it in the center of the room where
    # it would not see all of the points.
    cc = np.asarray([1, 1, 1])

    ###########################################################################
    # TODO: YOUR CAMERA CENTER CALCULATION CODE HERE
    ###########################################################################

    Q = M[:,:3]
    m4 = M[:,3]
    cc = -np.linalg.inv(Q).dot(m4)

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return cc

def estimate_fundamental_matrix(points_a, points_b):
    """
    Calculates the fundamental matrix. Try to implement this function as
    efficiently as possible. It will be called repeatedly in part 3.

    You must normalize your coordinates through linear transformations as
    described on the project webpage before you compute the fundamental
    matrix.

    Args:
    -   points_a: A numpy array of shape (N, 2) representing the 2D points in
                  image A
    -   points_b: A numpy array of shape (N, 2) representing the 2D points in
                  image B

    Returns:
    -   F: A numpy array of shape (3, 3) representing the fundamental matrix
    """

    # Placeholder fundamental matrix
    F = np.asarray([[0, 0, -0.0004],
                    [0, 0, 0.0032],
                    [0, -0.0044, 0.1034]])

    ###########################################################################
    # TODO: YOUR FUNDAMENTAL MATRIX ESTIMATION CODE HERE
    ###########################################################################

    def normalization_matrix_and_padding_ones(points):
        '''
        Args:
            points: (N,2) matrix
        Return:
            T: (3x3) matrix with both scale and mean normalization
            points_normalized_1s:  normalized (N,2) matrix after transformed by T
        '''
        cu,cv = np.mean(points, axis = 0) #(N,2) --axis=0-->(2,)
        average_square_normalized_distance = np.mean((points - np.array([cu,cv]))**2)     #(N,2) - (2,) = (N,2) --mean --> 1
        scale = np.sqrt(2.0/average_square_normalized_distance)

        scale_matrix = np.array([[scale, 0 ,0],[0,scale, 0],[0,0,1]])
        offset_matrix = np.array([[1, 0, -cu],[0, 1, -cv],[0, 0, 1]])
        T = scale_matrix.dot(offset_matrix)

        points_1s = np.ones([len(points),3])
        points_1s[:,0:2] = points

        points_normalized_1s = points_1s.dot(T.T) #(N,3).dot(3,3) = (N,3)

        return T, points_normalized_1s[:,0:2]
    
    Ta, points_normalized_a = normalization_matrix_and_padding_ones(points_a)
    Tb, points_normalized_b = normalization_matrix_and_padding_ones(points_b)
    
    #Solve the least square problem
    A = np.zeros([len(points_a), 8])  #8-points
    b = - np.ones([len(points_a), 1]) #f33 = 1, so flip to -1
    
    for ii in range(len(points_a)):
        ua, va = points_normalized_a[ii]
        ub, vb = points_normalized_b[ii]
        A[ii] = [ub*ua, va*ub, ub, ua*vb, va*vb, vb,ua,va]
        
    # Solve for M matrix using the least square
    F, residuals, rank, s = np.linalg.lstsq(A,b,rcond=None)  # F (8,1) matrix
    F = np.vstack([F,1.0]).reshape(3,3)                      # F (3,3) matrix
    F = Tb.T.dot(F).dot(Ta)                                  # Re-Normalized F with Tb, Ta
        
    # Impose the rank 2 conditions
    U, S, V = np.linalg.svd(F)
    S = np.array([[S[0], 0, 0],[0, S[1], 0],[0, 0, 0]])
    F = U.dot(S).dot(V)

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return F

def ransac_fundamental_matrix(matches_a, matches_b, confidence = 0.99):
    """
    Find the best fundamental matrix using RANSAC on potentially matching
    points. Your RANSAC loop should contain a call to
    estimate_fundamental_matrix() which you wrote in part 2.

    If you are trying to produce an uncluttered visualization of epipolar
    lines, you may want to return no more than 100 points for either left or
    right images.

    Args:
    -   matches_a: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from image A
    -   matches_b: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from image B
    Each row is a correspondence (e.g. row 42 of matches_a is a point that
    corresponds to row 42 of matches_b)

    Returns:
    -   best_F: A numpy array of shape (3, 3) representing the best fundamental
                matrix estimation
    -   inliers_a: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from image A that are inliers with
                   respect to best_F
    -   inliers_b: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from image B that are inliers with
                   respect to best_F
    """

    # Placeholder values
    best_F = estimate_fundamental_matrix(matches_a[:10, :], matches_b[:10, :])
    inliers_a = matches_a[:100, :]
    inliers_b = matches_b[:100, :]

    ###########################################################################
    # TODO: YOUR RANSAC CODE HERE
    ###########################################################################
    np.random.seed(2)
    
    N = len(matches_a)
    matches_1s_a = np.hstack([matches_a, np.ones([N,1])])  #(N,3)
    matches_1s_b = np.hstack([matches_b, np.ones([N,1])])  #(N,3)
    
    best_count = 0
    max_iter = 100*N
    threshold = 0.05
    RANSAC_FORCED_TO_STOP = False
    
    ii= 0
    while ii < max_iter:  # Will update the max_iter later
        ii += 1
        # Sample a random 10 points and calculate the F matrix
        random_sample = np.random.choice(N,8)
        points_a = matches_a[random_sample]
        points_b = matches_b[random_sample]
        
        temp_F = estimate_fundamental_matrix(points_a, points_b)
        
        # Calculate the number of inliner
        # First calculate Mb.T * F* Ma for all the N points
        score = matches_1s_b.dot(temp_F)   #(N,3).dot(3,3) = (N,3)
        score = np.sum(score*matches_1s_a, axis = 1)  #(N,3)*(N,3) = (N,3) --aixs=1--> (N,)
        
        # Second, choose a therehold for score as the inliners 
        # here I choose 0.01
        inliners = np.arange(N)[np.abs(score)<threshold]
        
        if len(inliners) > best_count:
            #print(np.mean(np.abs(score)))
            best_count = len(inliners)
            best_F, inliers_a, inliers_b = temp_F, matches_a[inliners], matches_b[inliners]
            
            # Update max_iter based on RANSAC formula
            p = best_count*1.0/N
            max_iter = np.log(1.0-confidence)/(np.log(1.0-p**6)-10**-8) # confidence = 0.95
            max_iter = min(200*N, max_iter) # No more than 100*N number of iterations
            
        if ii%(5*N) == 0:
            print('RANSAC is searching over %2dxlen(matches) iteration...' %(ii//N))
        
        if ii == 200*N:
            RANSAC_FORCED_TO_STOP = True
            
    if RANSAC_FORCED_TO_STOP:
        print('Stop RANSAC after 200 times the length of the matches set iteration')
    else:
        print('Early stopping! RANSAC completed after %d iteration' %ii)
        print('This equals to %.2f times the length of the matches set' %(ii*1.0/N))
    print('Achieves %d inliner over %d match points with threshold of %f ' %(best_count, N, threshold))
    
    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return best_F, inliers_a, inliers_b