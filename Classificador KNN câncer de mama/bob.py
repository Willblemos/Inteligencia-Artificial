from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn import tree
breast_cancer = load_breast_cancer()
print(breast_cancer.feature_names)
#print(diabetes.data_filename)
#print(diabetes.target_filename)
#print(diabetes.data[0])
#print(diabetes.target[0])

test_idx = [0, 50, 100]

#train data
train_target = np.delete(breast_cancer.target, test_idx)
train_data = np.delete(breast_cancer.data, test_idx, axis = 0)

#testing data
test_target = breast_cancer.target[test_idx]
test_data = breast_cancer.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)
print(test_target)
print(clf.predict(test_data))

#visualization code
from sklearn.externals.six import StringIO
import pydotplus
dot_data = StringIO()
tree.export_graphviz(clf, out_file = dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("breast_cancer.pdf")
