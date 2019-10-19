import cv2
import numpy as np
import pickle
from utils import load_image, load_image_gray
import cyvlfeat as vlfeat
import sklearn.metrics.pairwise as sklearn_pairwise
from sklearn.svm import LinearSVC
from IPython.core.debugger import set_trace
from utils import *

from sklearn.svm import SVC

from sklearn.cluster import KMeans
from sklearn.multiclass import OneVsRestClassifier  #for double-check only + extra credit
from collections import Counter


def get_tiny_images(image_paths):
    """
    This feature is inspired by the simple tiny images used as features in
    80 million tiny images: a large dataset for non-parametric object and
    scene recognition. A. Torralba, R. Fergus, W. T. Freeman. IEEE
    Transactions on Pattern Analysis and Machine Intelligence, vol.30(11),
    pp. 1958-1970, 2008. http://groups.csail.mit.edu/vision/TinyImages/

    To build a tiny image feature, simply resize the original image to a very
    small square resolution, e.g. 16x16. You can either resize the images to
    square while ignoring their aspect ratio or you can crop the center
    square portion out of each image. Making the tiny images zero mean and
    unit length (normalizing them) will increase performance modestly.

    Useful functions:
    -   cv2.resize
    -   use load_image(path) to load a RGB images and load_image_gray(path) to
      load grayscale images

    Args:
    -   image_paths: list of N elements containing image paths

    Returns:
    -   feats: N x d numpy array of resized and then vectorized tiny images
            e.g. if the images are resized to 16x16, d would be 256
    """
    # dummy feats variable
    feats = []

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################
    for img_path in image_paths:
        img = load_image_gray(img_path)  #np.array
        small = cv2.resize(img, (16,16)) #np.array
        feats.append(small.reshape(-1))

    feats = np.array(feats)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return feats

def build_vocabulary(image_paths, vocab_size):
    """
    This function will sample SIFT descriptors from the training images,
    cluster them with kmeans, and then return the cluster centers.

    Useful functions:
    -   Use load_image(path) to load RGB images and load_image_gray(path) to load
          grayscale images
    -   frames, descriptors = vlfeat.sift.dsift(img)
        http://www.vlfeat.org/matlab/vl_dsift.html
          -  frames is a N x 2 matrix of locations, which can be thrown away
          here (but possibly used for extra credit in get_bags_of_sifts if
          you're making a "spatial pyramid").
          -  descriptors is a N x 128 matrix of SIFT features
        Note: there are step, bin size, and smoothing parameters you can
        manipulate for dsift(). We recommend debugging with the 'fast'
        parameter. This approximate version of SIFT is about 20 times faster to
        compute. Also, be sure not to use the default value of step size. It
        will be very slow and you'll see relatively little performance gain
        from extremely dense sampling. You are welcome to use your own SIFT
        feature code! It will probably be slower, though.
    -   cluster_centers = vl_kmeans(X, K)
          http://www.vlfeat.org/matlab/vl_kmeans.html
            -  X is a N x d numpy array of sampled SIFT features, where N is
               the number of features sampled. N should be pretty large!
            -  K is the number of clusters desired (vocab_size)
               cluster_centers is a K x d matrix of cluster centers. This is
               your vocabulary.

    Args:
    -   image_paths: list of image paths.
    -   vocab_size: size of vocabulary

    Returns:
    -   vocab: This is a vocab_size x d numpy array (vocabulary). Each row is a
      cluster center / visual word
    """
    # Load images from the training set. To save computation time, you don't
    # necessarily need to sample from all images, although it would be better
    # to do so. You can randomly sample the descriptors from each image to save
    # memory and speed up the clustering. Or you can simply call vl_dsift with
    # a large step size here, but a smaller step size in get_bags_of_sifts.
    #
    # For each loaded image, get some SIFT features. You don't have to get as
    # many SIFT features as you will in get_bags_of_sift, because you're only
    # trying to get a representative sample here.
    #
    # Once you have tens of thousands of SIFT features from many training
    # images, cluster them with kmeans. The resulting centroids are now your
    # visual word vocabulary.

    dim = 128      # length of the SIFT descriptors that you are going to compute.
    vocab = np.zeros((vocab_size,dim))

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################
    
    # Loop over the images and randomly sample the descriptors
    stack_desciptors = []
    sample_size = int(20000//len(image_paths))  #take 20 000 sift features in total
    #print(sample_size,len(image_paths))
    
    for img_path in image_paths:
        img = load_image_gray(img_path)
        _, descriptors = vlfeat.sift.dsift(img, fast = True, step = 20)
        
        sample_indexes = np.random.permutation(len(descriptors))[:sample_size]
        sample_descriptors = descriptors[sample_indexes]
        
        stack_desciptors.append(sample_descriptors)
    
    stack_desciptors = np.array(stack_desciptors).reshape(-1, dim)
    
    # K_mean clustering to find the center
    kmeans = KMeans(n_clusters=vocab_size, random_state=0).fit(stack_desciptors)
    vocab = kmeans.cluster_centers_

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return vocab

def get_bags_of_sifts(image_paths, vocab_filename):
    """
    This feature representation is described in the handout, lecture
    materials, and Szeliski chapter 14.
    You will want to construct SIFT features here in the same way you
    did in build_vocabulary() (except for possibly changing the sampling
    rate) and then assign each local feature to its nearest cluster center
    and build a histogram indicating how many times each cluster was used.
    Don't forget to normalize the histogram, or else a larger image with more
    SIFT features will look very different from a smaller version of the same
    image.

    Useful functions:
    -   Use load_image(path) to load RGB images and load_image_gray(path) to load
          grayscale images
    -   frames, descriptors = vlfeat.sift.dsift(img)
          http://www.vlfeat.org/matlab/vl_dsift.html
        frames is a M x 2 matrix of locations, which can be thrown away here
          (but possibly used for extra credit in get_bags_of_sifts if you're
          making a "spatial pyramid").
        descriptors is a M x 128 matrix of SIFT features
          note: there are step, bin size, and smoothing parameters you can
          manipulate for dsift(). We recommend debugging with the 'fast'
          parameter. This approximate version of SIFT is about 20 times faster
          to compute. Also, be sure not to use the default value of step size.
          It will be very slow and you'll see relatively little performance
          gain from extremely dense sampling. You are welcome to use your own
          SIFT feature code! It will probably be slower, though.
    -   assignments = vlfeat.kmeans.kmeans_quantize(data, vocab)
          finds the cluster assigments for features in data
            -  data is a M x d matrix of image features
            -  vocab is the vocab_size x d matrix of cluster centers
            (vocabulary)
            -  assignments is a Mx1 array of assignments of feature vectors to
            nearest cluster centers, each element is an integer in
            [0, vocab_size)

    Args:
    -   image_paths: paths to N images
    -   vocab_filename: Path to the precomputed vocabulary.
          This function assumes that vocab_filename exists and contains an
          vocab_size x 128 ndarray 'vocab' where each row is a kmeans centroid
          or visual word. This ndarray is saved to disk rather than passed in
          as a parameter to avoid recomputing the vocabulary every run.

    Returns:
    -   image_feats: N x d matrix, where d is the dimensionality of the
          feature representation. In this case, d will equal the number of
          clusters or equivalently the number of entries in each image's
          histogram (vocab_size) below.
    """
    # load vocabulary
    with open(vocab_filename, 'rb') as f:
        vocab = pickle.load(f)

    # dummy features variable
    feats = []

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################
    
    for img_path in image_paths:
        img = load_image_gray(img_path)
        _, descriptors = vlfeat.sift.dsift(img, fast = True, step = 10)
        
        # Equivalent to K-mean center assignments:
        # First, Calculate the distance to the centers defined in vocab
        
        D = sklearn_pairwise.pairwise_distances(descriptors.astype('float64'), 
                                            vocab.astype('float64'), 
                                            metric = 'euclidean')   #(N,vocab_size)
        
        # Second, Assign label + bincount the label + normalize + append
        
        labels = np.argmin(D, axis = 1)  #(N,vocab_size) --axis=1--> (N,)
        
        hists = np.bincount(labels, minlength = len(vocab))  # minlength as len(vocab)
        hists = hists/np.linalg.norm(hists)
        
        feats.append(list(hists))

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return np.array(feats).astype('float64')

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats,
    metric='euclidean', k=1):
    """
    This function will predict the category for every test image by finding
    the training image with most similar features. Instead of 1 nearest
    neighbor, you can vote based on k nearest neighbors which will increase
    performance (although you need to pick a reasonable value for k).

    Useful functions:
    -   D = sklearn_pairwise.pairwise_distances(X, Y)
        computes the distance matrix D between all pairs of rows in X and Y.
          -  X is a N x d numpy array of d-dimensional features arranged along
          N rows
          -  Y is a M x d numpy array of d-dimensional features arranged along
          N rows
          -  D is a N x M numpy array where d(i, j) is the distance between row
          i of X and row j of Y

    Args:
    -   train_image_feats:  N x d numpy array, where d is the dimensionality of
          the feature representation
    -   train_labels: N element list, where each entry is a string indicating
          the ground truth category for each training image
    -   test_image_feats: M x d numpy array, where d is the dimensionality of the
          feature representation. You can assume N = M, unless you have changed
          the starter code
    -   metric: (optional) metric to be used for nearest neighbor.
          Can be used to select different distance functions. The default
          metric, 'euclidean' is fine for tiny images. 'chi2' tends to work
          well for histograms

    Returns:
    -   test_labels: M element list, where each entry is a string indicating the
          predicted category for each testing image
    """
    test_labels = []
    
    

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################

    D = sklearn_pairwise.pairwise_distances(test_image_feats.astype('float64'), 
                                            train_image_feats.astype('float64'), 
                                            metric = metric)   #(test,train)
    
    # Find out k-nearest neighbors
    nearest_neighbors_indexes = np.argsort(D, axis = 1)          #(test,train) --argsort axis=1--> (test,train)
    nearest_neighbors_indexes = nearest_neighbors_indexes[:,:k]  #(test,k) data = indexes
    nearest_neighbors_categories = np.array(train_labels)[nearest_neighbors_indexes]  # (test,k) data = categories labels
    
    # Need some helper dictionary
    
    if k==1:
        test_labels = nearest_neighbors_categories[:,0]
    else:
        for test_img in nearest_neighbors_categories:
            c = Counter(test_img)                     # count all the elements inside K- nearest neighbors
            this_categories = c.most_common()[0][0]   # choose the most frequent one
            test_labels.append(this_categories)       
            
        test_labels = np.array(test_labels)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return test_labels

def svm_classify(train_image_feats, train_labels, test_image_feats,
                 tol=1e-3, loss='squared_hinge', C=0.5):
    """
    This function will train a linear SVM for every category (i.e. one vs all)
    and then use the learned linear classifiers to predict the category of
    every test image. Every test feature will be evaluated with all 15 SVMs
    and the most confident SVM will "win". Confidence, or distance from the
    margin, is W*X + B where '*' is the inner product or dot product and W and
    B are the learned hyperplane parameters.

    Useful functions:
    -   sklearn LinearSVC
        http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
    -   svm.fit(X, y)
    -   set(l)

    Args:
    -   train_image_feats:  N x d numpy array, where d is the dimensionality of
          the feature representation
    -   train_labels: N element list, where each entry is a string indicating the
          ground truth category for each training image
    -   test_image_feats: M x d numpy array, where d is the dimensionality of the
          feature representation. You can assume N = M, unless you have changed
          the starter code
    Returns:
    -   test_labels: M element list, where each entry is a string indicating the
          predicted category for each testing image
    """
    # categories
    categories = list(set(train_labels))

    # construct 1 vs all SVMs for each category
    
    svms = {cat: LinearSVC(random_state=0, tol=tol, loss=loss, C=C) for cat in categories}

    test_labels = []

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################
    
    # Train SVM for each category
    W_svms, C_svms = [], []
    
    for cat in categories:
        # label 1 if img in cat else -1
        cat_labels = (np.array(train_labels) == cat)*2-1
        
        #fit the data
        svms[cat].fit(train_image_feats,cat_labels)
        
        #append the coefficient
        W_svms.append(svms[cat].coef_[0])          #(200,)
        C_svms.append(svms[cat].intercept_[0])     #float
        
    W_svms  = np.array(W_svms)   # (len(cat),200)
    C_svms  = np.array(C_svms)   # (len(cat),)
    
    # Label the test data
    
    # Calulate the pairwise distance from each test_image_feat to all the svms 
    SVM_distances = test_image_feats[:,None,:]*W_svms[None,:,:]  # (M,1,200)*(1,len(cat),200) =  (M,len(cat),200)
    SVM_distances = np.sum(SVM_distances, axis = -1)             # (M,len(cat),200) --axis=-1--> (M,len(cat))
    SVM_distances = SVM_distances + C_svms                       # (M,len(cat)) + (len(cat),) =  (M,len(cat))
    
    max_indexes = np.argmax(SVM_distances, axis = -1)            # (M,len(cat)) --axis=-1-->(M,)
    test_labels = np.array(categories)[max_indexes]              # (M,)
    
#     classifier = OneVsRestClassifier(LinearSVC(random_state=0,C=1))
#     classifier.fit(train_image_feats, train_labels)
#     test_labels = classifier.predict(test_image_feats)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return test_labels


########################################
###        EXTRA CODING              ###
########################################

def confusion_matrix_accuracy(predicted_categories,test_labels,categories):
    
    #Copy from the utils.py to have the same accuracy metric

    cat2idx = {cat: idx for idx, cat in enumerate(categories)}
    # confusion matrix
    y_true = [cat2idx[cat] for cat in test_labels]
    y_pred = [cat2idx[cat] for cat in predicted_categories]
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype(np.float) / cm.sum(axis=1)[:, np.newaxis]
    acc = np.mean(np.diag(cm))*100
    return acc


def svm_kernel(train_image_feats, train_labels, test_image_feats,
                 kernel = 'rbf', gamma = 0.1, C=1):

    classifier = OneVsRestClassifier(SVC(kernel=kernel, random_state=0, gamma=gamma, C=C))
    classifier.fit(train_image_feats, train_labels)
    test_labels = classifier.predict(test_image_feats)
    
    return test_labels

def kernel_codebook_encoding(image_paths, vocab_filename, gamma = 1):
    # load vocabulary
    with open(vocab_filename, 'rb') as f:
        vocab = pickle.load(f)

    # dummy features variable
    feats = []

    for img_path in image_paths:
        img = load_image_gray(img_path)
        _, descriptors = vlfeat.sift.dsift(img, fast = True, step = 10)
        
        # Equivalent to K-mean center assignments:
        # First, Calculate the distance to the centers defined in vocab
        
        D = sklearn_pairwise.pairwise_distances(descriptors.astype('float64'), 
                                            vocab.astype('float64'), 
                                            metric = 'euclidean')   #(N,vocab_size)
        
        # K(x,u) = exp(-gamma*(x-u)^2/2)
        D = np.exp(-gamma*0.5*D)            #(N,vocab_size)
        
        # Normalize
        D = D/np.sum(D, axis = 1)[:,None]   #(N,vocab_size)--axis=1--> (N,) -[:,None]-> (N,1) 
        
        # hist
        hists = np.sum(D, axis = 0)         #(N,vocab_size)--axis=0--> (vocab_size)
        hists = hists/np.linalg.norm(hists)
        
        feats.append(list(hists))

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return np.array(feats).astype('float64')
