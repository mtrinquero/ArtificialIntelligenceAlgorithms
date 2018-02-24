# Mark Trinquero
# Viterbi Algorithm Implimenation / Hidden Markov Models

    # RESOURCES CONSULTED:
    # https://en.wikipedia.org/wiki/Viterbi_algorithm#Example
    # https://en.wikipedia.org/wiki/Hidden_Markov_model
    # https://phvu.net/2013/12/06/sweet-implementation-of-viterbi-in-python/
    # https://en.wikipedia.org/wiki/Viterbi_algorithm
    # http://homepages.ulb.ac.be/~dgonze/TEACHING/viterbi.pdf
    # https://www.youtube.com/watch?v=RwwfUICZLsA
    # http://idiom.ucsd.edu/~rlevy/teaching/winter2009/ligncse256/lectures/hmm_viterbi_mini_example.pdf
    # http://www.cim.mcgill.ca/~latorres/Viterbi/va_alg.htm
    # https://web.stanford.edu/~jurafsky/slp3/8.pdf


# VITERBI ALGORITHM IMPLIMENTATION
# Method Inputs:
#   1) evidence_vector: A list of dictionaries mapping evidence variables to their values
#   2) prior: A dictionary corresponding to the prior distribution over states   
#   3) states: A list of all possible system states
#   4) evidence_variables: A list of all valid evidence variables
#   5) transition_probs: A dictionary mapping states onto dictionaries mapping states onto probabilities
#   6) emission_probs: A dictionary mapping states onto dictionaries mapping evidence variables onto probabilities for their possible values

def viterbi(evidence_vector, prior, states, evidence_variables, transition_probs, emission_probs): 
    if states != None:
        pass
    # assuming part of a larger distribution/sequence)
    if 'End' not in prior:
        prior['End'] = 0.0
    # storing paths for book keeping
    path_dict, V = [ {} ], [ {} ]
    for i in states: 
        path_dict[0][i] = [i]
        V[0][i] =  0  
        for j in states:
            for e in evidence_variables:  
                prob= (prior[j] * transition_probs[j][i] * emission_probs[i][e][evidence_vector[0][e]])
                if prob > V[0][i]: 
                    V[0][i] =  prob

    for t in range(1, len(evidence_vector)):
        path_dict.append( {} )
        V.append( {} )
        for i in states:
            path_dict[t][i] = []
            V[t][i] = 0
            for j in states:
                for e in evidence_variables:
                    prob = (V[t-1][j] * transition_probs[j][i] * emission_probs[i][e][evidence_vector[t][e]])
                    if prob >  V[t][i]:  
                        path_dict[t][i] = path_dict[t-1][j] + [i]
                        V[t][i] = prob  

    highest_prob = V[-1].values().index( max(V[-1].values() ))
    best = path_dict[-1].values()[highest_prob]
    most_likely_sequence = best
    return most_likely_sequence
    
