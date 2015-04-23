from sklearn import svm
import pandas as pd

training_data = pd.read_csv("training.csv")
headers = training_data.columns[1:10]
targets = list(training_data["OK"])

row_list = []
for ir, rr in training_data.iterrows():
    temp_list = []
    for val in training_data.columns[1:10]:
        temp_list.append(rr[val])
    row_list.append(temp_list)

X = row_list
y = targets
clf = svm.SVC()
clf.fit(X, y)

testing_data = pd.read_csv("test.csv")
new_row_list = []
for ir, rr in testing_data.iterrows():
    temp_list2 = []
    for val in testing_data.columns[1:10]:
        temp_list2.append(rr[val])
    new_row_list.append((rr["ID"], temp_list2))


for (x,y) in new_row_list:
    print(str(x) + "," + str(int(clf.predict(y))))
