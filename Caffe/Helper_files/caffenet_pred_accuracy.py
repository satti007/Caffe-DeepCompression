import glob
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

print 'Started Accuracy.......'

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cm=np.zeros([2,2],dtype='int')
cm[0][0]=1227
cm[0][1]=273
cm[1][0]=60
cm[1][1]=1440
accuracy=float(np.trace(cm))/3000
class_names=['cat','dog']
plt.figure()
plot_confusion_matrix(cm, classes=class_names,title='Confusion matrix,accuracy= %.1f'%(100*accuracy)+'%')
plt.show()



# def telldog(name):
# 	for synonym in dog_synonyms:
# 		if synonym in name:
# 			return True
# 	return False

# def iilegal(name):
# 	for label in iilegal_labels:
# 		if label in name:
# 			return True
# 	return False

# dog_synonyms=[]
# iilegal_labels=[]

# file = open('dog_synonyms.txt', 'r')
# for line in file:
# 	dog_synonyms.append(line.strip('\n'))

# file = open('iilegal_labels.txt', 'r')
# for line in file:
# 	iilegal_labels.append(line.strip('\n'))

# predict=[]
# lookup=[]
# with open('caffenet_predictions.txt','r') as f:
#     for line in f:
#         temp=line.split(' ', 1)[1].strip('\n').split(',')
#         for index,name in enumerate(temp):
#             name.lstrip()
#             if 'cat' in name:
#                 if iilegal(name):
#                     predict.append(-1)
#                 else:
#                     predict.append(0)
#                 break
#             elif telldog(name):
#         		predict.append(1)
#         		break
#             elif iilegal(name):
#         		predict.append(-1)
#         		break	
#             elif index == len(temp)-1:
#         		predict.append(temp[0])
#         		lookup.append(temp[0])
#         		break


# if(len(lookup)==0):
#     expected=[]
#     test_data = sorted([img for img in glob.glob("../input/test/*jpg")])
#     for idx,img_path in enumerate(test_data):
#     	if 'cat' in img_path:
#     		expected.append(0)
#     	else:
#     		expected.append(1)

#     print len(expected)
#     for index,label in enumerate(predict):
#     	if label== -1:
#             predict[index]=expected[index]^1


#     y_actu=expected
#     y_pred=predict
#     cm=confusion_matrix(y_actu, y_pred)
#     print np.trace(cm)
#     accuracy=float(np.trace(cm))/3000
#     class_names=['cat','dog']
#     plt.figure()
#     plot_confusion_matrix(cm, classes=class_names,title='Confusion matrix,accuracy= %f'%(100*accuracy)+'%')
#     plt.show()
# else:
#     lookup=set(lookup)
#     print 'lookup labels: ',len(lookup)
#     thefile = open('lookup.txt', 'w')
#     for item in lookup:
#       thefile.write("%s\n" % item)

