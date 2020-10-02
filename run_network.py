import numpy as np
import DigitIdentifier as di
import MNISTdataParser as mdp
import random


def grade_net(net: di.DigitIdentifier, test_output):
    """calculate the percentage of answers the network got right"""
    results = net.results
    right_answers = 0
    total = len(test_output)
    for k in range(len(results)):
        if results[k] == test_output[k]:
            right_answers += 1
    return 100 * (right_answers / total)


if __name__ == '__main__':
    training_images = mdp.get_training_inputs()
    training_results = mdp.get_training_outputs()
    training_data = mdp.organize_data(training_images, training_results)
    random.shuffle(training_data)
    
    test_images = mdp.get_test_inputs()
    test_results = mdp.get_test_outputs()
    test_data = mdp.organize_data(test_images, test_results)
    random.shuffle(test_data)

"""
    myNet = di.DigitIdentifier()
    myNet.back_prop_learning(training_data)
    myNet.run(test_data)

    print("accuracy: {:0.3f}%".format(myNet.accuracy))"""