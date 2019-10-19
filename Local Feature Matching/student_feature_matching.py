import numpy as np
from scipy.spatial.distance import cdist  # For optional part on bells and whistles


def match_features(features1, features2, x1, y1, x2, y2, metric='euclidean'):
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    For extra credit you can implement various forms of spatial/geometric
    verification of matches, e.g. using the x and y locations of the features.

    Args:
    -   features1: A numpy array of shape (n,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
    -   features2: A numpy array of shape (m,feat_dim) representing a second set
            features (m not necessarily equal to n)
    -   x1: A numpy array of shape (n,) containing the x-locations of features1
    -   y1: A numpy array of shape (n,) containing the y-locations of features1
    -   x2: A numpy array of shape (m,) containing the x-locations of features2
    -   y2: A numpy array of shape (m,) containing the y-locations of features2

    Returns:
    -   matches: A numpy array of shape (k,2), where k is the number of matches.
            The first column is an index in features1, and the second column is
            an index in features2
    -   confidences: A numpy array of shape (k,) with the real valued confidence for
            every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """
    #############################################################################
    # TODO: YOUR CODE HERE                                                        #
    #############################################################################
    
    match_pairs = []
    
    for idx1 in range(features1.shape[0]):
        # Broadcasting to calculate the distance to feature1[idx1]
        distance = np.linalg.norm(features1[idx1]-features2, axis = 1)

        # Find out the best and second best indexes
        idx2, idx2_second_best = np.argsort(distance)[:2]
        ratio = distance[idx2]/distance[idx2_second_best]
        
        # Follow the paper to only select the pair where ratio[best]< 0.8*ratio[second_best]
        if ratio<0.8:
            match_pairs.append([idx1, idx2, ratio])
            
    sorted_match_pairs = sorted(match_pairs, key=lambda x: x[2])
    sorted_match_pairs = np.array(sorted_match_pairs)
    
    matches = sorted_match_pairs[:,0:2]
    matches = matches.astype(int)
    confidences = sorted_match_pairs[:,2]
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return matches, confidences



###########################################################################################
# Below are codes for optional session. I implement the match features with for loop ######
###########################################################################################

def match_features_NAIVE(features1, features2, x1, y1, x2, y2, metric='euclidean'):
    '''
        Naive for loop to calculate the distance
    '''
    match_pairs = []
    for idx1 in range(features1.shape[0]):
        nearest_distance = float('inf')
        second_nearest_distance = float('inf')
        match_index = -1
        
        for idx2 in range(features2.shape[0]):
            current_distance = np.linalg.norm(features1[idx1]-features2[idx2])
            if current_distance < nearest_distance:
                nearest_distance, second_nearest_distance = current_distance, nearest_distance
                match_index = idx2
            elif current_distance < second_nearest_distance:
                second_nearest_distance = current_distance
        
        ratio = nearest_distance/second_nearest_distance
        
        # Follow the paper to only select the pair where ratio[best]< 0.8*ratio[second_best]
        if ratio<0.8:
            match_pairs.append([idx1, match_index, ratio])
     
    sorted_match_pairs = sorted(match_pairs, key=lambda x: x[2])
    sorted_match_pairs = np.array(sorted_match_pairs)
    matches = sorted_match_pairs[:,0:2]
    matches = matches.astype(int)
    confidences = sorted_match_pairs[:,2]
    return matches, confidences


def match_features_SCIPY(features1, features2, x1, y1, x2, y2, metric = 'euclidean'):
    '''
        SCIPY to calculate the distance matrix 
    '''
    match_pairs = []
    for idx1 in range(features1.shape[0]):
       
        #calculate the distances using SCIPY toolbox
        distance = cdist(features2, [features1[idx1]],
                          metric=metric)
        distance = np.ravel(distance)

        ########################
        
        idx2, idx2_second_best = np.argsort(distance)[:2]
        ratio = distance[idx2]/distance[idx2_second_best]
        
        # Follow the paper to only select the pair where ratio[best]< 0.8*ratio[second_best]
        if ratio<0.8:
            match_pairs.append([idx1, idx2, ratio])
    sorted_match_pairs = sorted(match_pairs, key=lambda x: x[2])
    sorted_match_pairs = np.array(sorted_match_pairs)
    
    matches = sorted_match_pairs[:,0:2]
    matches = matches.astype(int)
    confidences = sorted_match_pairs[:,2]
    return matches, confidences
    