import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances

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

def imbalance_impact_knn(dataset, labels):
    # the majority class is denoted as negative and the minority class as positive
    class_counts = torch.bincount(labels)
    majority_class = torch.argmax(class_counts).item()
    minority_class = torch.argmin(class_counts).item()
    majority_indices = (labels == majority_class).nonzero(as_tuple=True)[0]
    minority_indices = (labels == minority_class).nonzero(as_tuple=True)[0]
    selected_indices = torch.cat((majority_indices, minority_indices))
    data = dataset[selected_indices].numpy()
    label = labels[selected_indices].numpy()
    label[labels[selected_indices] == majority_class] = -1
    label[labels[selected_indices] == minority_class] = 1

    pos_num = sum(label == 1)
    neg_num = sum(label == -1)
    pos_idx = np.nonzero(label == 1)
    neg_idx = np.nonzero(label == -1)
    pos_data = data[pos_idx]
    rr = neg_num / pos_num
    k = 5

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(data)
    distances, knn_idx = nbrs.kneighbors(pos_data)

    p2 = np.zeros(pos_num)
    p2old = np.zeros(pos_num)
    knn_idx = np.delete(knn_idx, 0, 1)
    for i in range(pos_num):
        p2[i] = np.intersect1d(knn_idx[i], neg_idx).size / k
        p2old[i] = p2[i]
        if p2[i] == 1:
            dist = pairwise_distances(pos_data[i].reshape(1, -1), data).reshape(-1)
            sort_idx = np.argsort(dist)
            nearest_pos = np.nonzero(label[sort_idx] == 1)[0][1]
            p2[i] = (nearest_pos - 1) / nearest_pos
    p1 = 1 - p2
    # ibi3 = p2 * rr * p1 / (p2 + rr * p1)
    # ibi3 = p2 * (rr * p1 / (p2 + rr * p1) - p1)
    ibi3 = (rr * p1 / (p2 + rr * p1) - p1)

    bi3 = np.mean(ibi3)
    print('The bi3 value is %.4f' % bi3)

if __name__ == "__main__":
    print("size: ", len(dataset.labels))
    labels = dataset.labels
    Imbalance_factor(labels)
    Gini(labels)
    LT_Ratio(labels, 0.8)
    CCDF(labels)
    imbalance_impact_knn(dataset.features, labels)


