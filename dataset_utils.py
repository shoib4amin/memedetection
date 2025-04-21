import pickle
from csv import reader
from tqdm import tqdm



def read_csv(file_name):
    feat_lst = []
    labels = []
    with open(file_name, 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Iterate over each row in the csv using reader object
        for row in tqdm(csv_reader):
            # row variable is a list that represents a row in csv
            class_label = row[-1]
            # print(class_label)
            # exit()
            row = row[:-1]
            s_feat = [float(i) for i in row]
            feat_lst.append(s_feat)
            labels.append(class_label)
        return feat_lst, labels
