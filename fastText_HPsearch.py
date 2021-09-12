import fasttext
import sys
import argparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--train', action="store", dest='train',
                        required=True,
                        help='the path of train data set')
    parser.add_argument('-d', '--dev', action="store", dest='dev',
                        required=True,
                        help='the path of dev dataset')
    parser.add_argument('-te', '--test', action="store", dest='test',
                        required=True,
                        help='the path of test dataset')
    args = parser.parse_args()
    model = fasttext.train_supervised(input = args.train, autotuneValidationFile = args.dev, autotuneDuration=600)

    for threshold in [0.5]:
      prediction = []
      labels = []
      for line in open(args.dev):
        label, text = line.split('\t')
        #pre =  model.predict(text.replace("\n"," "))[0][0]
        pre =  model.predict(text.replace("\n"," "))[0][0]
        #prob =  model.predict(text.replace("\n"," "))[1][0]
        if pre == '__label__biased': # and prob >= threshold: 
          prediction.append(1)
        else: 
          prediction.append(0)
        # prediction.append(random.choice(k))
        if label == '__label__biased': 
          labels.append(1)
        if label == '__label__neutral': 
          labels.append(0)

      f1 = f1_score(labels, prediction, pos_label = 1)
      precision = precision_score(labels, prediction, pos_label = 1)
      recall = recall_score(labels, prediction, pos_label = 1)
      print("  Dev set:")
      print("  The F1 score is {:}".format(f1))
      print("  The precision is {:}".format(precision))
      print("  The recall is {:}\n".format(recall))

      for threshold in [0.5]:
        prediction = []
        labels = []
        for line in open(args.test):
          label, text = line.split('\t')
          #pre =  model.predict(text.replace("\n"," "))[0][0]
          pre =  model.predict(text.replace("\n"," "))[0][0]
          #prob =  model.predict(text.replace("\n"," "))[1][0]
          if pre == '__label__biased': # and prob >= threshold: 
            prediction.append(1)
          else: 
            prediction.append(0)
          # prediction.append(random.choice(k))
          if label == '__label__biased': 
            labels.append(1)
          if label == '__label__neutral': 
            labels.append(0)

        f1 = f1_score(labels, prediction, pos_label = 1)
        precision = precision_score(labels, prediction, pos_label = 1)
        recall = recall_score(labels, prediction, pos_label = 1)
        print("  Test set:")
        print("  The F1 score is {:}".format(f1))
        print("  The precision is {:}".format(precision))
        print("  The recall is {:}\n".format(recall))

if __name__ == "__main__":
    main(sys.argv[1:])