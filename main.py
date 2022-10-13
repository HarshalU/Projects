from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

kv = 3

irisData = datasets.load_iris()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(irisData.data, irisData.target, test_size=0.25, random_state=2603)
classifier = KNeighborsClassifier(kv)
classifier.fit(X_train, y_train)

n1 = []
n2 = []
pos = 0
neg = 0
for i in range(kv):
    n1.append(999999)
    n2.append(-1)

labels = set(y_train)
target = {}
for i in labels:
    target[i] = 0

for i, test_data in enumerate(X_test):

    n3 = list(n1)
    n4 = list(n2)

    for j, point2 in enumerate(X_train):

        distance = 0
        for axis in range(len(test_data)):
            distance += (test_data[axis] - point2[axis]) ** 2
        distance = distance ** 1 / 2

        for k in range(kv):
            if n3[k] > distance:
                n3.insert(k, distance)
                n3.pop()
                n4.insert(k, j)
                n4.pop()
                break

    targetdic = dict(target)

    nearest = np.take(y_train, n4)

    for j in nearest:
        targetdic[j] += 1

    m1 = -1
    m2 = -1
    for j in targetdic.keys():
        if m2 < targetdic[j]:
            m1 = j
            m2 = targetdic[j]

    if m1 == y_test[i]:
        pos += 1
    else:
        neg += 1

error = []
for i in range(1, 40):
    knn = KNeighborsClassifier(i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(20, 10))
plt.plot(range(1, 40), error, color='green', marker='o', markerfacecolor='yellow', markersize=5)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

print("no of errors: " + str(neg))
print("% of accurancy: " + str(pos / (pos + neg) * 100))

def calculate_distance(X_tr, y_tr):
    train_length = X_tr.shape[0]
    same_class_dist, other_class_dist = [[math.inf for i in range(train_length)] for j in range(2)]

    for i in range(train_length-1):
        for j in range(i+1,train_length):
            distance = np.linalg.norm(X_tr[i]-X_tr[j])

            if y_tr[i]==y_tr[j]:
                if distance < same_class_dist[i]:
                    same_class_dist[i] = distance
                if distance < same_class_dist[j]:
                    same_class_dist[j] = distance
            else:
                if distance < other_class_dist[i]:
                    other_class_dist[i] = distance
                if distance < other_class_dist[j]:
                    other_class_dist[j] = distance

    return [same_class_dist, other_class_dist]
def conformal_prediction(X, y, dataset, test_size=0.3, train_size=0.7, random_state=2603):
    predicted_list, p_values = [[] for i in range(2)]
    # splitting the data
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=test_size, train_size=train_size, random_state=random_state)
    lenrange = len(list(set(y_train)))
    same_class_dist, other_class_dist = calculate_distance(X_train, y_train)

    for i in range(len(X_test)):
        conformity_scores = [[] for j in range(lenrange)]
        curr_testXval = X_test[i]
        for j in range(lenrange):
            new_same_dist = np.append(same_class_dist, math.inf)
            new_other_class_dist = np.append(other_class_dist, math.inf)
            extended_X = np.concatenate((X_train, [curr_testXval]), axis = 0)
            extended_y = np.concatenate((y_train, [j]), axis = 0)

            for curr_idx, curr_elem in enumerate(extended_X):
                distance = np.linalg.norm(curr_elem - curr_testXval)
                idx = len(extended_X)-1

                if distance != 0: #to avoid duplicate value
                    if j == extended_y[curr_idx]:
                        if distance < new_same_dist[idx]:
                            new_same_dist[idx] = distance
                    else:
                        if distance < new_other_class_dist[idx]:
                            new_other_class_dist[idx] = distance

                if new_same_dist[curr_idx] == 0: #to avoid duplicate value
                    conformity_scores[j].append(0)
                else:
                    conformity_scores[j].append(new_other_class_dist[curr_idx]/new_same_dist[curr_idx])

        p_vals = []
        for k in range(lenrange):
            p_vals.append(np.mean(conformity_scores[k]<=conformity_scores[k][X_train.shape[0]]))

        predicted_list.append(p_vals.index(max(p_vals)))
        p_values.append(p_vals)

    falsep = []
    for i, p in enumerate(p_values):
        sumval = 0;
        for j, q in enumerate(p):
            if j != y_test[i]:
                sumval += q
        falsep.append(sumval)

    false_p_value = np.sum(falsep)/(len(falsep)*2)
    accuracy = np.mean(predicted_list == y_test)

    print(
          "The average false p-value : {} \n"
          "The accuracy of prediction : {} \n"
          "The test error rate is : {}"
          .format(false_p_value, accuracy, 1-accuracy))

conformal_prediction(irisData.data, irisData.target, irisData, test_size=0.3, train_size=0.7, random_state=2603)