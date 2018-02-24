# Mark Trinquero
# Reinforcement Learning - Decision Trees / Binary Classification
# Advanced Decision Tree Learning

import math
import numpy as np


####--------------------------------------------------
#### K-Folds Validation: training and testing   
####--------------------------------------------------

def load_csv(data_file_path, class_index=-1):
    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])
    classes= map(int,  out[:,class_index])
    features = out[:, :class_index]
    return features, classes


def generate_k_folds(dataset, k):
    # Note: each fold is a tuple like (training_set, test_set) where each set is a tuple like (examples, classes)
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


####--------------------------------------------------
#### K-Folds Validation: training and testing for k=10 example 
####--------------------------------------------------
# dataset = load_csv('part2_data.csv')
# ten_folds = generate_k_folds(dataset, 10)
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

#     TESTING OUTPUT - 98%
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

