from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)

from sklearn.datasets import load_iris

iris = load_iris()
X_train,y_train = iris.data,iris.target
tree.fit(X_train, y_train)

from sklearn.tree import export_graphviz

export_graphviz(tree, out_file='treeIRISgini.dot', feature_names=['sepal length', 'sepal width','petal length','petal width'])