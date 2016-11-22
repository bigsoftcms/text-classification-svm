import numpy as np
from sklearn import svm
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.cross_validation import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
import argparse
from scipy.sparse.csr import csr_matrix

def load_sparse_matrix(input_file_path):

    data_row = [];
    data_col = [];

    data_value = [];

    input_file = open(input_file_path, 'r')

    for l in input_file.readlines():
    	ll = l.rstrip().split(',')
    	data_row.append(ll[0])
    	data_col.append(int(ll[1]))
    	data_value.append(float(ll[2]))

    row_list = []
    for i in xrange(len(data_row)):
    	if data_row[i] not in row_list:
    		row_list.append(data_row[i])	
    	data_row[i] = int(row_list.index(data_row[i]))

    return csr_matrix(( data_value, (data_row, data_col)  ) ), row_list


def classifyPatents(feature_vectors, labels):
    
    test_proportion = 0.3
    labels = label_binarize(labels,classes= ['A','B','C','D','E','F','G','H'])
    random_state = np.random.RandomState(0)
    X_train, X_test, y_train, y_test = train_test_split(feature_vectors, labels, test_size = test_proportion,
                                                        random_state=random_state)
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    n_classes = 8
    # Compute Precision-Recall and plot curve
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

    # Compute micro-average ROC curve and ROC area
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
        y_score.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_score,
                                                         average="micro")
    
        
    print 'AUC={0:0.2f}'.format(average_precision['micro'])


parser = argparse.ArgumentParser(description='get AUC score')

parser.add_argument('feature_vector_file', help = 'file path of feature vectors')

parser.add_argument('label_file', help = 'file path of labels')

args = parser.parse_args()

if __name__ == '__main__':
    
    label_file = open(args.label_file, 'r')

    labels = label_file.readlines()

    label_ids = [l.rstrip().split(',')[0] for l in labels]
    
    o_labels = [l.rstrip().split(',')[1] for l in labels]

    feature_vectors,row_list = load_sparse_matrix(args.feature_vector_file)

    n_labels = []

    for i in xrange(len(row_list)):
       n_labels.append(o_labels[label_ids.index(row_list[i])])
    
    
    
    classifyPatents(feature_vectors, n_labels)    
    
    label_file.close()
        
    





