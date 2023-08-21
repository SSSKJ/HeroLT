import numpy as np
import matplotlib.pyplot as plt

def Imbalance_factor(labels):
    num_labels = labels.max() + 1
    num_labels_each_class = np.array([(labels == i).sum().item() for i in range(num_labels)])
    num_labels_each_class = np.array([i for i in num_labels_each_class if i > 0])
    sorted_num_labels_each_class = np.sort(num_labels_each_class)[::-1]

    IF = sorted_num_labels_each_class[0]/sorted_num_labels_each_class[-1]
    print("categories: ", num_labels.item())
    print("Imbalance factor: %.3f" % IF)

def Gini(labels):
    labels = np.array(labels, dtype=np.int64)
    total = 0
    for i, xi in enumerate(labels[:-1], 1):
        total += np.sum(np.abs(xi - labels[i:]))
    gini = total / (len(labels) ** 2 * np.mean(labels))
    print("Gini coefficient: %.3f" % gini)

def LT_Ratio(labels, alpha):
    num_labels = labels.max() + 1
    num_labels_each_class = np.array([(labels == i).sum().item() for i in range(num_labels)])
    sorted_num_labels_each_class = np.sort(num_labels_each_class)[::-1]

    sum_1, sum_2 = 0, 0
    for i in range(num_labels):
        if sum_1 >= alpha * len(labels):
            break
        else:
            sum_1 += sorted_num_labels_each_class[i]

    for j in range(num_labels):
        if sum_2 >= (1-alpha) * len(labels):
            break
        else:
            sum_2 += sorted_num_labels_each_class[num_labels-j-1]
    print("Long-Tailedness Ratio: %.3f" % (i/j))

def CCDF(labels):
    fontSize = 15
    labels = np.array(labels, dtype=np.int64)
    sorted_data = np.sort(labels)
    ccdf = 1.0 - np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    # Plot in log-log scale
    plt.figure(figsize=(8, 6))
    plt.loglog(sorted_data, ccdf, linewidth=3, label='Actual CCDF')
    plt.xlabel('X (log scale)', fontsize=fontSize)
    plt.ylabel('CCDF (log scale)', fontsize=fontSize)
    # plt.title('Complementary Cumulative Distribution Function', fontsize=fontSize)
    plt.grid(True)
    plt.legend(fontsize=fontSize)
    plt.show()

if __name__ == "__main__":
    print("size: ", len(dataset.labels))
    labels = dataset.labels
    Imbalance_factor(labels)
    Gini(labels)
    LT_Ratio(labels, 0.8)
    CCDF(labels)


