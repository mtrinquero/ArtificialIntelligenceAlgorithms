# Mark Trinquero
# Reinforcement Learning - Decision Trees / Binary Classification
# Advanced Decision Tree Learning

import math
import numpy


class DecisionNode():
    def __init__(self, left, right, decision_function,class_label=None):
        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        if self.class_label is not None:
            return self.class_label        
        return self.left.decide(feature) if self.decision_function(feature) else self.right.decide(feature)



####--------------------------------------------------
#### Decision Tree Learning    
####--------------------------------------------------


def entropy(class_vector):
    # http://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python
    # https://en.wikipedia.org/wiki/Entropy_(information_theory) 
    class_vector_list = list(class_vector)
    pos = class_vector_list.count(1)
    neg = class_vector_list.count(0)
    class_len = len(class_vector_list)
    if pos == 0:
        q = 0
    elif class_len == 0:
        q = 0
    else:
        q = pos / float(class_len)
    if q == 0:  # log base 2 of 1 = 0
        return 0.00000000000001
    elif q == 1:
        return 0.00000000000001
    else:
        q_inv = 1 - q
        return -q * np.log2(q) - q_inv * np.log2(q_inv)



def information_gain(previous_classes, current_classes):    
    # https://en.wikipedia.org/wiki/Information_gain_in_decision_trees
    positive_classes = []
    negative_classes = []
    for i in range(len(current_classes)):
        if current_classes[i] == 1:
            positive_classes.append(previous_classes[i])
        elif current_classes[i] == 0:
            negative_classes.append(previous_classes[i])
        else:
            print "Error in information_gain function: class wasa not recognized"
    positive_subset = len(positive_classes)/ float(len(current_classes))* entropy(positive_classes)
    negative_subset = len(negative_classes)/ float(len(current_classes))* entropy(negative_classes)
    return entropy(previous_classes) - (negative_subset+positive_subset)


class DecisionTree():

    def __init__(self, depth_limit=float('inf')):
        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):
        # https://en.wikipedia.org/wiki/C4.5_algorithm
        if len(set(classes)) == 1:
            return DecisionNode(None, None, None, classes[0])
        if depth >= self.depth_limit:
            leaf_output = max(set(classes), key = list(classes).count)
            return DecisionNode(None, None, None, leaf_output)
        alpha_best = None
        gain_best = -float('inf')
        # For each attribute alpha: evaluate the normalized information gain gained by splitting on alpha
        for alpha_i in range(len(features[0])):
            #split values - set alpha as step value for indexing
            alpha_features = features[:, alpha_i]
            #set threshold to mean of alpha values subset
            threshold = alpha_features.mean()
            new_alpha_set = [1 if f >= threshold else 0 for f in alpha_features]
            # left (true) if above threshold, right (false) if below threshold
            IG = information_gain(classes, new_alpha_set)
            # Let alpha_best be the attribute with the highest normalized information gain
            # if current IG is better, set best alpha index
            if IG > gain_best:
                gain_best = IG
                alpha_best = alpha_i

        L_node = None
        R_node = None
        L_features, L_classes = [], []
        R_features, R_classes = [], []
        alpha_best_list = features[:, alpha_best]
        alpha_best_mean = alpha_best_list.mean()

        for i in range(len(features)):
            if features[i, alpha_best] >= alpha_best_mean:
                L_features.append(features[i])
                L_classes.append(classes[i])
            elif features[i, alpha_best] < alpha_best_mean:
                R_features.append(features[i])
                R_classes.append(classes[i])
            else:
                print "Error in __build_tree__: unexpexted alpha value"

        # Create a decision node that splits on alpha_best
        # Recur on the sublists obtained by splitting on alpha_best, and add those nodes as children of node
        # LEFT CHILDREN NODE(s) SET UP
        if len(L_features) > 0:
            L_node = self.__build_tree__(np.array(L_features), np.array(L_classes), depth + 1)
        # RIGHT CHILDREN NODE(s) SET UP
        if len(R_features) > 0:
            R_node = self.__build_tree__(np.array(R_features), np.array(R_classes), depth + 1)

        # Create a Decision node that splits on alpha_best (w/ mean value as threshold)
        alpha_node = DecisionNode(L_node, R_node, lambda x: x[alpha_best] >= alpha_best_mean, None)

        print "-"*50
        print alpha_node.left.class_label
        print alpha_node.right.class_label
        print "-"*50
        print L_features
        print L_classes
        print "-"*50

        return alpha_node




# Use fitted decision tree to classify list of feature vectors
    def classify(self, features):
        output = []
        fitted_tree = self.root
        for feature in features:
            temp = fitted_tree.decide(feature)
            temp = int(temp)
            output.append(temp)
        return output



####--------------------------------------------------
####        PART 2B - Validation  
####--------------------------------------------------
# For this part of the assignment we're going to use a relatively simple dataset (banknote authentication), 
# found in 'part_2_data.csv'. In the section below there are methods to load the data in a consistent format.

# In general, reserving part of your data as a test set can lead to unpredictable performance- a serendipitous choice 
# of your train or test split could give you a very inaccurate idea of how your classifier performs. 
# That's where k-fold cross validation comes in.

# In the below method, we'll split the dataset at random into k equal subsections, then iterating on each of our k samples, 
# we'll reserve that sample for testing and use the other k-1 for training. Averaging the results of each fold should give 
# us a more consistent idea of how the classifier is doing.


import numpy as np
def load_csv(data_file_path, class_index=-1):
    # Class index is last column (E) in test data file..
    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])
    classes= map(int,  out[:,class_index])
    features = out[:, :class_index]
    return features, classes





import math
def generate_k_folds(dataset, k):
    # TODO this method should return a list of folds,
    # where each fold is a tuple like (training_set, test_set)
    # where each set is a tuple like (examples, classes)
    features = dataset[0]
    classes = list(dataset[1])

    rng_state = np.random.get_state()
    np.random.shuffle(classes)
    np.random.set_state(rng_state)
    np.random.shuffle(features)

    subset_size = int(math.ceil(len(features) / k))

    folds = []

    for i in range(k):
        test_start = i * subset_size
        test_end = i * subset_size + subset_size

        train_start_L = 0
        train_end_L = i * subset_size

        train_start_R = i * subset_size + subset_size + 1
        train_end_R = -1

        # TEST DATA SETUP
        test_feats = features[test_start:test_end]
        test_classes = classes[test_start:test_end]
        test = (test_feats, test_classes)

        # TRAIN DATA SETUP
        train_feats = np.concatenate((features[train_start_L:train_end_L], features[train_start_R:train_end_R]), axis=0)
        train_classes = np.concatenate((classes[train_start_L:train_end_L], classes[train_start_R:train_end_R]), axis=0)
        train = (train_feats, train_classes)

        # K FOLD DATA ((,),(,))
        folds.append((train, test))

    return folds





# def generate_k_folds(dataset, k):
#     #TODO this method should return a list of folds,
#     # where each fold is a tuple like (training_set, test_set)
#     # where each set is a tuple like (examples, classes)
#     X = dataset[0]
#     y = dataset[1]

#     num_folds = k
#     folds = []
#     subset_size = int(len(X) / k)

#     for n in range(num_folds):

#         Xarray_slice = X[n * subset_size: (n+1)* subset_size]
#         yarray_slice = y[n * subset_size: (n+1)* subset_size]

#         X_valid = Xarray_slice
#         y_valid = yarray_slice

#         X_train = np.delete(X, Xarray_slice)
#         y_train = np.delete(y, yarray_slice)

#         training_set = (X_train, y_train)
#         test_set = (X_valid, y_valid)

#         fold = (training_set, test_set)
#         folds.append(fold)

#     return folds




# def generate_k_folds(dataset, k):
#     #TODO this method should return a list of folds,
#     # where each fold is a tuple like (training_set, test_set)
#     # where each set is a tuple like (examples, classes)

#     # dataset = features, classes
#     #   - features: X (data)
#     #   - classes: y (class lables)

#     #features, classes = dataset
#     #X, y = dataset
#     X = dataset[0]
#     y = dataset[1]

#     # print "X = ", X
#     # print "Y = ", y 

#     num_folds = k
#     folds = []
#     subset_size = int(len(X) / k)

#     for n in range(num_folds):

#         Xarray_slice = X[n * subset_size: (n+1)* subset_size]
#         yarray_slice = y[n * subset_size: (n+1)* subset_size]

#         X_valid = Xarray_slice
#         y_valid = yarray_slice

#         X_train = np.delete(X, Xarray_slice)
#         y_train = np.delete(y, yarray_slice)

#         training_set = (X_train, y_train)
#         test_set = (X_valid, y_valid)

#         fold = (training_set, test_set)
#         folds.append(fold)


#         #X_train = np.concatenate( (X[:n * subset_size]), (X[(n + 1) * subset_size:]) )

#         # X_train_1 = X[:n * subset_size]
#         # X_train_2 = X[(n + 1) * subset_size:]

#         # print X_train_1
#         # print X_train_2

#         # tot = X_train_1.append(X_train_2)
#         # print tot

#         # Xarray_slice = X[n * subset_size: (n+1)* subset_size]
#         # yarray_slice = y[n * subset_size: (n+1)* subset_size]


#         #X_train = np.concatenate( (X[:n * subset_size]), (X[(n + 1) * subset_size:]) )

#         #y_train = np.concatenate( (y[:n * subset_size]), (y[(n + 1) * subset_size:]) ) 

#         # X_valid = X[n * subset_size: (n+1)* subset_size]
#         # y_valid = y[n * subset_size: (n+1)* subset_size]

#         # training_set = (X_train, y_train)
#         # test_set = (X_valid, y_valid)

#         # fold = (training_set, test_set)
#         #folds.append(fold)

#     return folds







# dataset = load_csv('part2_data.csv')


# # print "#"*75
# # print " ----  DATASET  ----"
# # print dataset
# # print "#"*75


# ten_folds = generate_k_folds(dataset, 10)


# #on average your accuracy should be higher than 60%.
# accuracies = []
# precisions = []
# recalls = []
# confusion = []

# for fold in ten_folds:
#     train, test = fold
#     train_features, train_classes = train
#     test_features, test_classes = test
#     tree = DecisionTree( )
#     tree.fit( train_features, train_classes)

#     #tree.fit( test_features, test_classes)

#     # TESTING OUTPUT - 98%
#     output = tree.classify(test_features)
#     accuracies.append( accuracy(output, test_classes))
#     precisions.append( precision(output, test_classes))
#     recalls.append( recall(output, test_classes))
#     confusion.append( confusion_matrix(output, test_classes))

    # TRAINING OUTPUT - 100%
    # output = tree.classify(train_features)
    # accuracies.append( accuracy(output, train_classes))
    # precisions.append( precision(output, train_classes))
    # recalls.append( recall(output, train_classes))
    # confusion.append( confusion_matrix(output, train_classes))



# # 2A & 2B TESTING
# print "#"*50
# print "2A and 2B Testing Output"
# print "#"*50
# print "     P2 Accuracy List:    ", accuracies
# print "-"*50
# print "     P2 Precision List:   ", precisions
# print "-"*50
# print "     P2 Recall List:      ", recalls
# print "-"*50
# print "     P2 C_Matrix List:    ", confusion
# print "-"*50



















####--------------------------------------------------
####        PART 3 - 30 pts - Random Forests    
####--------------------------------------------------
# The decision boundaries drawn by decision trees are very sharp, and fitting a decision tree of unbounded depth to a list of 
# examples almost inevitably leads to overfitting. In an attempt to decrease the variance of our classifier we're going to 
# use a technique called 'Bootstrap Aggregating' (often abbreviated 'bagging').

# A Random Forest is a collection of decision trees, built as follows:

# 1) For every tree we're going to build:

#     a) Subsample the examples provided (with replacement) in accordance with a example subsampling rate.     
#     b) From the sample in a), choose attributes at random to learn on, in accordance with an attribute subsampling rate.
#     c) Fit a decision tree to the subsample of data we've chosen (to a certain depth)

# Note : The 'example subsampling rate' and 'attribute subsampling rate' are to be passed as arguments as shown below.
#     
# Classification for a random forest is then done by taking a majority vote of the classifications 
# yielded by each tree in the forest after it classifies an example.

import random

class RandomForest():

    def __init__(self, num_trees, depth_limit, example_subsample_rate, attr_subsample_rate):
        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        # TODO implement the above algorithm to build a random forest of decision trees

        subsample_rate = self.example_subsample_rate
        attr_subsample_rate = self.attr_subsample_rate
        num_attr_to_accept = int(len(features[0]) * attr_subsample_rate)

        num_feat_to_accept = int(len(features) * subsample_rate)
        num_class_to_accept = int(len(classes) * subsample_rate)

        #ensure generate state remains the same for shuffle
        rng_state = np.random.get_state()
        np.random.shuffle(classes)
        np.random.set_state(rng_state)
        np.random.shuffle(features)

        #keep % of shuffled classed to keep for building tree based on subsample rate
        new_features = list(features[:num_feat_to_accept])
        new_classes = list(classes[:num_class_to_accept])

        #randomly select attributes to learn on from above 
        to_learn = []
        random_atts = []

        for att in range(num_attr_to_accept):
            ri = random.randint(0,len(features[0])-1)
            random_atts.append(ri)


        for att in random_atts:
            tmp = new_features[:, att]
            to_learn.append(tmp)


#     c) Fit a decision tree to the subsample of data we've chosen (to a certain depth)
        tree = DecisionTree()
        tree.fit(to_learn, new_classes)
        return tree






    def classify(self, features):
        # TODO implement classification for a random forest.




        raise NotImplemented()








#TODO: As with the DecisionTree, evaluate the performance of your RandomForest on the dataset for part 2.
# on average your accuracy should be higher than 75%.

#  Optimize the parameters of your random forest for accuracy for a forest of 5 trees.
# (We'll verify these by training one of your RandomForest instances using these parameters
#  and checking the resulting accuracy)

#  Fill out the function below to reflect your answer:

def ideal_parameters():
    # TODO - this is just a stub for a total guess 
    ideal_depth_limit = 3
    ideal_esr = 4
    ideal_asr = 5

    return ideal_depth_limit, ideal_esr, ideal_asr











































####--------------------------------------------------
####        PART 4 - 10 pts - Challenge!    
####--------------------------------------------------
# You've been provided with a sample of data from a research dataset in 'challenge_data.pickle'. 
# It is serialized as a tuple of (features, classes). I have reserved a part of the dataset for testing. 
# The classifier that performs most accurately on the holdout set wins (so optimize for accuracy). 
# To get full points for this part of the assignment, you'll need to get at least an average accuracy of 80% 
# on the data you have (testing on training data), and at least an average accuracy of 60% on the holdout set (test data set).

# Ties will be broken by submission time.
# First place:  3 bonus points on your final grade
# Second place: 2 bonus points on your final grade
# Third place:  1 bonus point on your final grade



class ChallengeClassifier():
    
    def __init__(self):
        # initialize whatever parameters you may need here-
        # this method will be called without parameters 
        # so if you add any to make parameter sweeps easier, provide defaults
        raise NotImplemented()
        


    def fit(self, features, classes):
        # fit your model to the provided features
        raise NotImplemented()


        
    def classify(self, features):
        # classify each feature in features as either 0 or 1.
        raise NotImplemented()



        
