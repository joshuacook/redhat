from numpy import mean

def precision_and_recall(prediction, actual):
    condition_1 = (actual == True)
    condition_2 = (actual == False)
    corr_true = sum(prediction[condition_1] == True)
    incr_true = sum(prediction[condition_2] == True)
    corr_fals = sum(prediction[condition_2] == False)
    incr_fals = sum(prediction[condition_1] == False)

    return corr_true, incr_true, corr_fals, incr_fals, \
            corr_true/float(corr_true+incr_fals), \
            corr_true/float(corr_true+incr_true)

def measure_accuracy(prediction, actual):
    return mean(prediction == actual)

def measure_f1_score(prediction, actual):
    _, _, _, _, precision, recall = precision_and_recall(prediction, actual)
    try:
        return 2/(1/float(recall) + 1/float(precision))
    except ZeroDivisionError:
        return 0


def correlation_matrix(prediction, actual):
    corr_true, incr_true, corr_fals, incr_fals, _, _ = \
            precision_and_recall(prediction, actual)

    print(
    """
                    act      act
                   true    false   totals
    +------------+-------+-------+-------+
    | pred true  | {:5} | {:5} | {:5} |
    +------------+-------+-------+-------+
    | pred false | {:5} | {:5} | {:5} |
    +------------+-------+-------+-------+
    |  totals    | {:5} | {:5} | {:5} |
    +------------+-------+-------+-------+
    """.format(
        corr_true,
        incr_true,
        corr_true+incr_true,
        incr_fals,
        corr_fals,
        incr_fals+corr_fals,
        corr_true+incr_fals,
        incr_true+corr_fals,
        corr_true+incr_fals+incr_true+corr_fals
        )
    )
