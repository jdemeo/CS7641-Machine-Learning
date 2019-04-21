import time

def training_or_testing(data, network, start_time):
    '''TRAINING'''

    # Store error metrics for each iteration
    correct = 0
    incorrect = 0
    error = 0.00

    for instance in data:
        network.setInputValues(instance.getData())
        network.run()

        # ACCURACY
        predicted = network.getOutputValues().get(0)
        actual = instance.getLabel().getContinuous()

        if abs(predicted - actual) < 0.5:
            correct += 1
        else:
            incorrect += 1

    end = time.time()
    the_time = end - start_time

    accuracy = float(correct)/(correct+incorrect)*100.0

    print "Training or Test"
    print "Accuracy: %0.03f" % accuracy

    return the_time, accuracy
