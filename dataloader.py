import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import numpy as np


class IEMOCAPDataset_BERT4(Dataset):

    def __init__(self, path, train=True):

        (
            self.videoIDs,
            self.videoSpeakers,
            self.videoLabels,
            self.videoText,
            self.videoAudio,
            self.videoVisual,
            self.videoSentence,
            self.trainVid,
            self.testVid,
        ) = pickle.load(open(path, "rb"), encoding="latin1")
        self.keys = self.trainVid if train else self.testVid

        self.len = len(self.keys)

        self.labels_emotion = self.videoLabels

        labels_sentiment = {}
        for item in self.videoLabels:
            array = []
            for e in self.videoLabels[item]:
                if e in [1, 3]:
                    array.append(0)
                elif e == 2:
                    array.append(1)
                elif e in [0]:
                    array.append(2)
            labels_sentiment[item] = array
        self.labels_sentiment = labels_sentiment

    def __getitem__(self, index):
        vid = self.keys[index]
        return (
            torch.FloatTensor(np.array(self.videoText[vid])),
            torch.FloatTensor(np.array(self.videoText[vid])),
            torch.FloatTensor(np.array(self.videoText[vid])),
            torch.FloatTensor(np.array(self.videoText[vid])),
            torch.FloatTensor(np.array(self.videoVisual[vid])),
            torch.FloatTensor(np.array(self.videoAudio[vid])),
            torch.FloatTensor(
                [
                    [1, 0] if x == "M" else [0, 1]
                    for x in np.array(self.videoSpeakers[vid])
                ]
            ),
            torch.FloatTensor([1] * len(np.array(self.labels_emotion[vid]))),
            torch.LongTensor(np.array(self.labels_emotion[vid])),
            torch.LongTensor(np.array(self.labels_sentiment[vid])),
            vid,
        )

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        return [
            (
                pad_sequence(dat[i])
                if i < 7
                else pad_sequence(dat[i]) if i < 10 else dat[i].tolist()
            )
            for i in dat
        ]


class IEMOCAPDataset_BERT(Dataset):

    def __init__(self, path, train=True):

        (
            self.videoIDs,
            self.videoSpeakers,
            self.videoLabels,
            self.videoText0,
            self.videoText1,
            self.videoText2,
            self.videoText3,
            self.videoAudio,
            self.videoVisual,
            self.videoSentence,
            self.trainVid,
            self.testVid,
        ) = pickle.load(open(path, "rb"), encoding="latin1")

        self.keys = self.trainVid if train else self.testVid

        self.len = len(self.keys)

        self.labels_emotion = self.videoLabels

        labels_sentiment = {}
        for item in self.videoLabels:
            array = []
            for e in self.videoLabels[item]:
                if e in [1, 3, 5]:
                    array.append(0)
                elif e == 2:
                    array.append(1)
                elif e in [0, 4]:
                    array.append(2)
            labels_sentiment[item] = array
        self.labels_sentiment = labels_sentiment

    def __getitem__(self, index):
        vid = self.keys[index]
        return (
            torch.FloatTensor(np.array(self.videoText0[vid])),
            torch.FloatTensor(np.array(self.videoText1[vid])),
            torch.FloatTensor(np.array(self.videoText2[vid])),
            torch.FloatTensor(np.array(self.videoText3[vid])),
            torch.FloatTensor(np.array(self.videoVisual[vid])),
            torch.FloatTensor(np.array(self.videoAudio[vid])),
            torch.FloatTensor(
                [
                    [1, 0] if x == "M" else [0, 1]
                    for x in np.array(self.videoSpeakers[vid])
                ]
            ),
            torch.FloatTensor([1] * len(np.array(self.labels_emotion[vid]))),
            torch.LongTensor(np.array(self.labels_emotion[vid])),
            torch.LongTensor(np.array(self.labels_sentiment[vid])),
            vid,
        )

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        return [
            (
                pad_sequence(dat[i])
                if i < 7
                else pad_sequence(dat[i]) if i < 10 else dat[i].tolist()
            )
            for i in dat
        ]


class MELDDataset_BERT(Dataset):

    def __init__(self, path, train=True):
        """
        label index mapping = {0:neutral, 1:surprise, 2:fear, 3:sadness, 4:joy, 5:disgust, 6:anger}
        """
        (
            self.videoIDs,
            self.videoSpeakers,
            self.videoLabels,
            self.videoSentiments,
            self.videoText0,
            self.videoText1,
            self.videoText2,
            self.videoText3,
            self.videoAudio,
            self.videoVisual,
            self.videoSentence,
            self.trainVid,
            self.testVid,
            _,
        ) = pickle.load(open(path, "rb"))

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

        self.labels_emotion = self.videoLabels

        self.labels_sentiment = self.videoSentiments

    def __getitem__(self, index):
        vid = self.keys[index]
        return (
            torch.FloatTensor(np.array(self.videoText0[vid])),
            torch.FloatTensor(np.array(self.videoText1[vid])),
            torch.FloatTensor(np.array(self.videoText2[vid])),
            torch.FloatTensor(np.array(self.videoText3[vid])),
            torch.FloatTensor(np.array(self.videoVisual[vid])),
            torch.FloatTensor(np.array(self.videoAudio[vid])),
            torch.FloatTensor(np.array(self.videoSpeakers[vid])),
            torch.FloatTensor([1] * len(np.array(self.labels_emotion[vid]))),
            torch.LongTensor(np.array(self.labels_emotion[vid])),
            torch.LongTensor(np.array(self.labels_sentiment[vid])),
            vid,
        )

    def __len__(self):
        return self.len

    def return_labels(self):
        return_label = []
        for key in self.keys:
            return_label += self.videoLabels[key]
        return return_label

    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        return [
            (
                pad_sequence(dat[i])
                if i < 7
                else pad_sequence(dat[i]) if i < 10 else dat[i].tolist()
            )
            for i in dat
        ]


class CMUMOSEIDataset7(Dataset):

    def __init__(self, path, train=True):

        (
            self.videoIDs,
            self.videoSpeakers,
            self.videoLabels,
            self.videoText,
            self.videoAudio,
            self.videoVisual,
            self.videoSentence,
            self.trainVid,
            self.testVid,
        ) = pickle.load(open(path, "rb"), encoding="latin1")

        self.keys = self.trainVid if train else self.testVid

        self.len = len(self.keys)

        labels_emotion = {}
        for item in self.videoLabels:
            array = []
            for a in self.videoLabels[item]:
                if a < -2:
                    array.append(0)
                elif -2 <= a and a < -1:
                    array.append(1)
                elif -1 <= a and a < 0:
                    array.append(2)
                elif 0 <= a and a <= 0:
                    array.append(3)
                elif 0 < a and a <= 1:
                    array.append(4)
                elif 1 < a and a <= 2:
                    array.append(5)
                elif a > 2:
                    array.append(6)
            labels_emotion[item] = array
        self.labels_emotion = labels_emotion

        labels_sentiment = {}
        for item in self.videoLabels:
            array = []
            for a in self.videoLabels[item]:
                if a < 0:
                    array.append(0)
                elif 0 <= a and a <= 0:
                    array.append(1)
                elif a > 0:
                    array.append(2)
            labels_sentiment[item] = array
        self.labels_sentiment = labels_sentiment

    def __getitem__(self, index):
        vid = self.keys[index]
        return (
            torch.FloatTensor(np.array(self.videoText[vid])),
            torch.FloatTensor(np.array(self.videoText[vid])),
            torch.FloatTensor(np.array(self.videoText[vid])),
            torch.FloatTensor(np.array(self.videoText[vid])),
            torch.FloatTensor(np.array(self.videoVisual[vid])),
            torch.FloatTensor(np.array(self.videoAudio[vid])),
            torch.FloatTensor(
                [
                    [1, 0] if x == "M" else [0, 1]
                    for x in np.array(self.videoSpeakers[vid])
                ]
            ),
            torch.FloatTensor([1] * len(np.array(self.labels_emotion[vid]))),
            torch.LongTensor(np.array(self.labels_emotion[vid])),
            torch.LongTensor(np.array(self.labels_sentiment[vid])),
            vid,
        )

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        return [
            (
                pad_sequence(dat[i])
                if i < 7
                else pad_sequence(dat[i]) if i < 10 else dat[i].tolist()
            )
            for i in dat
        ]
