import numpy as np
import matplotlib.pyplot as plt

def experiment_1(n1, n2, mean1, mean2, cov1, cov2):
    data_A1 = np.random.multivariate_normal(mean1, cov1, n1)
    data_A2 = np.random.multivariate_normal(mean2, cov2, n2)

    # Plot dataset A clusters
    plt.scatter(data_A1[:, 0], data_A1[:, 1], color='blue')
    plt.scatter(data_A2[:, 0], data_A2[:, 1], color='red')

    # Problem 1A
    '''
    We will be using a case I because the covariance matrix is the same and our matrices have no covariances
    g(x) = wi^tx + wi0

    where wi = ui / s^2 and 
    '''

    p_w1 = n1 / (n1 + n2)  # Prior Probability of sample 1
    p_w2 = n2 / (n1 + n2)  # Prior Probability of sample 2

    print(f'(p_w1, p_w2) = ({p_w1}, {p_w2})')

    # Find means
    mu_x1, mu_x2 = np.mean(data_A1[:, 0]), np.mean(data_A2[:, 0])
    mu_y1, mu_y2 = np.mean(data_A1[:, 1]), np.mean(data_A2[:, 1])
    print(f'(mu_x1, mu_y1) = ({mu_x1}, {mu_y1})\n(mu_x2, mu_y2) = ({mu_x2}, {mu_y2})')
    mu_1 = np.array([mu_x1, mu_y1])
    mu_2 = np.array([mu_x2, mu_y2])
    print(f"mu1: {mu_1}\nmu2: {mu_2}")
    plt.scatter(mu_1[0], mu_1[1], color = 'green')
    plt.scatter(mu_2[0], mu_2[1], color='green')

    # Find stdev
    std_x1, std_x2 = np.std(data_A1[:, 0]), np.std(data_A2[:, 0])
    std_y1, std_y2 = np.mean(data_A1[:, 1]), np.mean(data_A2[:, 1])
    std_1 = np.array([std_x1, std_y1])
    std_2 = np.array([std_x2, std_y2])
    print(f"std1: {std_1}\nstd2: {std_2}")

    # Calculate x0
    # x0 = .5 * (mu_x1 + mu_x2) - (cov1[0][0] / (mu_x1 - mu_x2)**2) * np.log(p_w1 / p_w2) * (mu_x1 - mu_x2)
    # y0 = .5 * (mu_y1 + mu_y2) - (cov1[1][1] / (mu_y1 - mu_y2)**2) * np.log(p_w1 / p_w2) * (mu_y1 - mu_y2)
    x0 = .5 * (mu_x1 + mu_x2) - (std_x1**2 / np.abs(mu_x1 - mu_x2)) * np.log((p_w1) / (p_w2)) * (mu_x1 - mu_x2)
    y0 = .5 * (mu_y1 + mu_y2) - (std_y1**2 / np.abs(mu_y1 - mu_y2)) * np.log((p_w1) / (p_w2)) * (mu_y1 - mu_y2)

    print(f'(x0, y0) = ({x0},{y0})')
    plt.scatter(x0, y0, color='green')
    x0_term = np.array([x0, y0])

    # Calculate w
    w = mu_1 - mu_2
    w = w.reshape((2, 1))
    print(f'w = {w}')

    # Cluster A1 classification
    yhat_A1 = np.dot(w.T, (data_A1 - x0_term).T)
    print(f'Cluster A1:\n{yhat_A1}\n\nA1 Matrix:\n{data_A1}\n')

    # Cluster A2 classification
    yhat_A2 = np.dot(w.T, (data_A2 - x0).T)
    print(f'Cluster A2:\n{yhat_A2}\n\nA2 Matrix:\n{data_A2}\n')

    # Classify TP, TN, FP, FN
    TP_counter, TN_counter, FP_counter, FN_counter = 0, 0, 0, 0
    A1_len, A2_len = len(yhat_A1), len(yhat_A2)
    # Checking Positives
    for y_hat in yhat_A1.flatten():
        print(y_hat)
        # TP Condition
        if y_hat >= 0:
            TP_counter += 1
        # FP Condition
        else:
            FP_counter += 1
    # Checking Negatives
    for y_hat in yhat_A2.flatten():
        print(y_hat)
        # TN Condition
        if y_hat <= 0:
            TN_counter += 1
        # FN Condition
        else:
            FN_counter += 1
    print(f'TP Rate: {TP_counter}')
    print(f'FP Rate: {FP_counter}')
    print(f'TN Rate: {TN_counter}')
    print(f'FN Rate: {FN_counter}\n')

    # Plot decision boundary on scatter plot
    wi = (1 / std_1 * mu_1).reshape((2, 1))
    print(f'Std: \n{std_1}, {std_1**2}')
    print(f'Mu: \n{mu_1}')
    print(f'wi - \n{wi}')
    wi0 = -1 / (2 * std_1 ** 2) * np.dot(mu_1.T, mu_1) + np.log(p_w1) # CHANGE TO P(wi)
    print(f'wi0 - \n{wi0}')
    print("TEST")
    x, y = [], []
    for i,j in data_A2:
        x_val = wi[0] * i + wi0[0]
        y_val = -1 * wi[1] * i + wi0[1]
        x.append(x_val)
        y.append(y_val)
        print(f'({i}, {wi[0]}, {x_val}, {wi[1]}, {y_val})')
    plt.scatter(x,y, color = 'orange')
    for i,j in data_A1:
        x_val = wi[0] * i + wi0[0]
        y_val = -1 * wi[1] * i + wi0[1]
        x.append(x_val)
        y.append(y_val)
        print(f'({i}, {wi[0]}, {x_val}, {wi[1]}, {y_val})')
    plt.scatter(x,y, color = 'orange')
    plt.savefig('mygraph.png')

def experiment_2(data_A1, data_A2, n1, n2, cov1, cov2):

    print("\n**EXPERIMENT 2 CLASSIFIER**\n")

    # Plot dataset A clusters
    plt.scatter(data_A1[:, 0], data_A1[:, 1], color='blue')
    plt.scatter(data_A2[:, 0], data_A2[:, 1], color='red')

    # Find cov
    cov1 = np.array([[np.std(data_A1[:, 0])**2, 0], [0,np.std(data_A1[:, 1])**2]]).reshape((2,2))
    cov2 = np.array([[np.std(data_A2[:, 0])**2, 0], [0, np.std(data_A2[:, 1])**2]]).reshape((2,2))

    ## TESTS
    #cov1 = np.array([[.5, 0], [0, 2]])
    #cov2 = np.array([[2,0], [0,2]])
    ##

    print(f"cov1: \n{cov1}\ncov2: \n{cov2}")

    # Prior Probabilities
    p_w1 = n1 / (n1 + n2)  # Prior Probability of sample 1
    p_w2 = n2 / (n1 + n2)  # Prior Probability of sample 2
    print(f'(p_w1, p_w2) = ({p_w1}, {p_w2})\n')

    cov1_inv = np.linalg.inv(cov1)
    cov2_inv = np.linalg.inv(cov2)

    print(f'cov1_inv:\n{cov1_inv}\ncov2_inv:\n{cov2_inv}\n')

    W1 = (-1/2) * cov1_inv
    W2 = (-1/2) * cov2_inv
    print(f"W1: \n{W1}\nW2: \n{W2}\n")

    mu_1 = np.array([np.mean(data_A1[:, 0]), np.mean(data_A1[:, 1])]).reshape((2,1))
    mu_2 = np.array([np.mean(data_A2[:, 0]), np.mean(data_A2[:, 1])]).reshape((2,1))

    ## TESTS
    #mu_1 = np.array([3,6]).reshape((2,1))
    #mu_2 = np.array([3,-2]).reshape((2, 1))
    ##

    plt.scatter(mu_1[0], mu_1[1], color = 'green')
    plt.scatter(mu_2[0], mu_2[1], color='green')
    print(f"mu1: \n{mu_1}\nmu2: \n{mu_2}\n")

    w1 = np.dot(cov1_inv, mu_1)
    w2 = np.dot(cov2_inv, mu_2)
    print(f"w1: \n{w1}\nw2: \n{w2}\n")

    w10 = np.dot(np.dot((-1/2) * mu_1.T, cov1_inv), mu_1) - (.5 * np.log(np.linalg.det(cov1))) + np.log(p_w1)
    w20 = np.dot(np.dot((-1/2) * mu_2.T, cov2_inv), mu_2) - (.5 * np.log(np.linalg.det(cov2))) + np.log(p_w2)
    print(f"w10: \n{w10}\nw20: \n{w20}\n")

    ## TESTS
    #data_A1 = np.array([[0,0], [1,1], [5, 5]]).reshape((3,2))
    #data_A2 = np.array([[0, 0], [1, 1], [5, 5]]).reshape((3, 2))
    ##

    g1 = np.dot(np.dot(data_A1, W1), data_A1.T) + np.dot(w1.T, data_A1.T) + w10
    g1 = np.diag(g1) # Extract g1 values from diagonal
    print(f'g1:{g1}\n')

    g2 = np.dot(np.dot(data_A2, W2), data_A2.T) + np.dot(w2.T, data_A2.T) + w20
    g2 = np.diag(g2)  # Extract g1 values from diagonal
    print(f'g2:{g2}\n')

    # Calculate new quadratic equation result for cluster A1
    g_diff = np.dot(np.dot(data_A1, (W1-W2)), data_A1.T) + np.dot((w1-w2).T, data_A1.T) + (w10-w20)
    #print(f'Factors: \nW:\n{W1-W2}\nw:\n{w1-w2}\n')
    g_diff_A1 = np.diag(g_diff)
    print(f'g_diff:{g_diff_A1}\n')

    # Calculate new quadratic equation result for cluster A2
    g_diff = np.dot(np.dot(data_A2, (W1-W2)), data_A2.T) + np.dot((w1-w2).T, data_A2.T) + (w10-w20)
    #print(f'Factors: \nW:\n{W1-W2}\nw:\n{w1-w2}\n')
    g_diff_A2 = np.diag(g_diff)
    print(f'g_diff:{g_diff_A2}\n')

    # Performance Statistics
    TP_counter, FP_counter, TN_counter, FN_counter = 0,0,0,0
    for val in g_diff_A1:
        # g_diff should be > 0 for dataset A1
        if val > 0:
            TP_counter += 1
        else:
            FP_counter += 1
    for val in g_diff_A2:
        # g_diff should be < 0 for dataset A2
        if val < 0:
            TN_counter += 1
        else:
            FN_counter += 1

    print(f'TP Rate: {TP_counter}')
    print(f'FP Rate: {FP_counter}')
    print(f'TN Rate: {TN_counter}')
    print(f'FN Rate: {FN_counter}\n')

    # Misclassification Rate
    misclassification_rate = (FP_counter + FN_counter) / (n1 + n2)
    print(f'Misclassification Rate: {misclassification_rate * 100}%')

    # Plot equation of line
    x = np.linspace(-0,5)
    #y = (wi_diff[0] * x + wi0_diff[0]) / -wi_diff[1]
    W_diff = W1-W2
    w_diff = w1-w2
    w_coeff_diff = w10-w20
    print(f'test\n{W_diff}\n{w_diff}\n{w_coeff_diff}')
    print(f'Equation:\n{W_diff[0][0]}x1^2 + {W_diff[1][1]}x2^2 + {w_diff[0][0]}x1 + {w_diff[1][0]}x2 + {w_coeff_diff[0][0]} = 0')
    y = (-W_diff[0][0]*(x**2) - w_diff[0][0]*x - w_coeff_diff[0][0] - w_diff[1][0]) / W_diff[1][1]
    #y = (-W_diff[0][0] * (x ** 2) - w_diff[0][0] * x - w_coeff_diff[0][0])
    '''
    w1 = a
    w2 = b
    w10 = c
    w20 = d
    '''
    plt.plot(x,y)


    # Bhattacharyya Bound
    term1 = np.dot(np.dot((1/8) * (mu_2 - mu_1).T, np.linalg.inv(.5 * cov1 + .5 * cov2)), (mu_2 - mu_1))
    numerator = np.linalg.det(.5 * cov1 + .5 * cov2)
    denominator = (np.linalg.det(cov1) * np.linalg.det(cov2)) ** .5
    k = term1 + .5 * np.log(numerator / denominator)
    p_error = 2 * (p_w1 * p_w2)**.5 * np.e**(-1 * k)
    print(f'p_error - {p_error}')

    plt.title("Quadratic Discriminant Function (Case III)")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.savefig('experiment2.png')


def experiment_3(data_A1, data_A2, n1, n2, cov1, cov2):
    # Euclidean distance classifier\
    print("\n**EXPERIMENT 3 CLASSIFIER**\n")

    # Plot dataset A clusters
    plt.scatter(data_A1[:, 0], data_A1[:, 1], color='blue')
    plt.scatter(data_A2[:, 0], data_A2[:, 1], color='red')

    print(f'DataA1 Shape - {data_A1.shape}\nDataA2 Shape - {data_A2.shape}\n')

    # Find cov
    cov1 = np.array([[np.std(data_A1[:, 0])**2, 0], [0,np.std(data_A1[:, 1])**2]]).reshape((2,2))
    cov2 = np.array([[np.std(data_A2[:, 0])**2, 0], [0, np.std(data_A2[:, 1])**2]]).reshape((2,2))

    # Prior Probabilities
    p_w1 = .5  # Prior Probability of sample 1
    p_w2 = .5  # Prior Probability of sample 2
    print(f'(p_w1, p_w2) = ({p_w1}, {p_w2})\n')

    # Find mu
    mu_1 = np.array([np.mean(data_A1[:, 0]), np.mean(data_A1[:, 1])]).reshape((2,1))
    mu_2 = np.array([np.mean(data_A2[:, 0]), np.mean(data_A2[:, 1])]).reshape((2,1))
    plt.scatter(mu_1[0], mu_1[1], color = 'green')
    plt.scatter(mu_2[0], mu_2[1], color='green')
    print(f"mu1: \n{mu_1}\nmu2: \n{mu_2}\n")\

    # Find stdev
    std = cov1[0][0]

    # Find wi
    w1 = mu_1 / (std**2)
    print(f'w1:\n{w1}')
    w2 = mu_2 / (std**2)
    print(f'w2:\n{w2}\n')

    # Find wi0
    w10 = (-1/(2 * std**2)) * np.dot(mu_1.T,mu_1) + np.log(p_w1)
    print(f'w10:\n{w10}')
    w20 = (-1/(2 * std**2)) * np.dot(mu_2.T,mu_2) + np.log(p_w2)
    print(f'w20:\n{w20}\n')

    # Find g
    g1 = np.dot(w1.T, data_A1.T) + w10
    print(f'g1: \n{g1}')
    g2 = np.dot(w2.T, data_A2.T) + w20
    print(f'g2: \n{g2}')

    # Plot lines
    g_diff_A1 = np.dot((w1-w2).T, data_A1.T) + w10 - w20
    print(f'Factors:\n{w1-w2}\n{w10-w20}')
    print(f'g_diff_A1: \n{g_diff_A1}')
    g_diff_A2 = np.dot((w1-w2).T, data_A2.T) + w10 - w20
    print(f'g_diff_A2: \n{g_diff_A2}')
    wi_diff = w1-w2
    wi0_diff = w10-w20

    x = np.linspace(-0,5)
    y = (wi_diff[0] * x + wi0_diff[0]) / -wi_diff[1]
    plt.plot(x,y)

    # Performance Statistics
    TP_counter, FP_counter, TN_counter, FN_counter = 0,0,0,0
    for val in g_diff_A1[0]:
        if val > 0:
            TP_counter += 1
        else:
            FP_counter += 1
    for val in g_diff_A2[0]:
        if val < 0:
            TN_counter += 1
        else:
            FN_counter += 1
    print(f'TP Rate: {TP_counter}')
    print(f'FP Rate: {FP_counter}')
    print(f'TN Rate: {TN_counter}')
    print(f'FN Rate: {FN_counter}\n')

    # Misclassification Rate
    misclassification_rate = (FP_counter + FN_counter) / (n1 + n2)
    print(f'Misclassification Rate: {misclassification_rate * 100}%')

    # Bhattacharyya Bound
    term1 = np.dot(np.dot((1/8) * (mu_2 - mu_1).T, np.linalg.inv(.5 * cov1 + .5 * cov2)), (mu_2 - mu_1))
    numerator = np.linalg.det(.5 * cov1 + .5 * cov2)
    denominator = (np.linalg.det(cov1) * np.linalg.det(cov2)) ** .5
    k = term1 + .5 * np.log(numerator / denominator)
    p_error = 2 * (p_w1 * p_w2)**.5 * np.e**(-1 * k)
    print(f'p_error - {p_error}')

    plt.savefig('experiment3.png')

def experiment_1pt2(data_A1, data_A2, n1, n2, cov1, cov2):
    print("\n**EXPERIMENT 1 CLASSIFIER**\n")
    # Plot dataset A clusters
    plt.scatter(data_A1[:, 0], data_A1[:, 1], color='blue')
    plt.scatter(data_A2[:, 0], data_A2[:, 1], color='red')

    print(f'DataA1 Shape - {data_A1.shape}\nDataA2 Shape - {data_A2.shape}\n')

    # Find cov
    cov1 = np.array([[np.std(data_A1[:, 0])**2, 0], [0,np.std(data_A1[:, 1])**2]]).reshape((2,2))
    cov2 = np.array([[np.std(data_A2[:, 0])**2, 0], [0, np.std(data_A2[:, 1])**2]]).reshape((2,2))

    # Prior Probabilities
    p_w1 = n1 / (n1 + n2)  # Prior Probability of sample 1
    p_w2 = n2 / (n1 + n2)  # Prior Probability of sample 2
    print(f'(p_w1, p_w2) = ({p_w1}, {p_w2})\n')

    # Find mu
    mu_1 = np.array([np.mean(data_A1[:, 0]), np.mean(data_A1[:, 1])]).reshape((2,1))
    mu_2 = np.array([np.mean(data_A2[:, 0]), np.mean(data_A2[:, 1])]).reshape((2,1))
    plt.scatter(mu_1[0], mu_1[1], color = 'green')
    plt.scatter(mu_2[0], mu_2[1], color='green')
    print(f"mu1: \n{mu_1}\nmu2: \n{mu_2}\n")\

    # Find stdev
    std = cov1[0][0]

    # Find wi
    w1 = mu_1 / (std**2)
    print(f'w1:\n{w1}')
    w2 = mu_2 / (std**2)
    print(f'w2:\n{w2}\n')

    # Find wi0
    w10 = (-1/(2 * std**2)) * np.dot(mu_1.T,mu_1) + np.log(p_w1)
    print(f'w10:\n{w10}')
    w20 = (-1/(2 * std**2)) * np.dot(mu_2.T,mu_2) + np.log(p_w2)
    print(f'w20:\n{w20}\n')

    # Find g
    g1 = np.dot(w1.T, data_A1.T) + w10
    print(f'g1: \n{g1}')
    g2 = np.dot(w2.T, data_A2.T) + w20
    print(f'g2: \n{g2}')

    # Plot lines
    g_diff_A1 = np.dot((w1-w2).T, data_A1.T) + w10 - w20
    print(f'Factors:\n{w1-w2}\n{w10-w20}')
    print(f'g_diff_A1: \n{g_diff_A1}')
    g_diff_A2 = np.dot((w1-w2).T, data_A2.T) + w10 - w20
    print(f'g_diff_A2: \n{g_diff_A2}')
    wi_diff = w1-w2
    wi0_diff = w10-w20

    x = np.linspace(-0,5)
    y = (wi_diff[0] * x + wi0_diff[0]) / -wi_diff[1]
    plt.plot(x,y)

    # Performance Statistics
    TP_counter, FP_counter, TN_counter, FN_counter = 0,0,0,0
    for val in g_diff_A1[0]:
        if val > 0:
            TP_counter += 1
        else:
            FP_counter += 1
    for val in g_diff_A2[0]:
        if val < 0:
            TN_counter += 1
        else:
            FN_counter += 1
    print(f'TP Rate: {TP_counter}')
    print(f'FP Rate: {FP_counter}')
    print(f'TN Rate: {TN_counter}')
    print(f'FN Rate: {FN_counter}\n')

    # Misclassification Rate
    misclassification_rate = (FP_counter + FN_counter) / (n1 + n2)
    print(f'Misclassification Rate: {misclassification_rate * 100}%')

    # Bhattacharyya Bound
    term1 = np.dot(np.dot((1/8) * (mu_2 - mu_1).T, np.linalg.inv(.5 * cov1 + .5 * cov2)), (mu_2 - mu_1))
    numerator = np.linalg.det(.5 * cov1 + .5 * cov2)
    denominator = (np.linalg.det(cov1) * np.linalg.det(cov2)) ** .5
    k = term1 + .5 * np.log(numerator / denominator)
    p_error = 2 * (p_w1 * p_w2)**.5 * np.e**(-1 * k)
    print(f'p_error - {p_error}')

    plt.savefig('experiment1pt2.png')
