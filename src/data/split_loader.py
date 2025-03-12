'''
Since only the left training images have associated bounding box labels,
the training set is split to allow validation of the model
'''


from src.data.data_loader import ImageLoader


class TrainLoader(ImageLoader):
    def __init__(self, dataDir, split, transform, fusion_level):
        super().__init__(dataDir, split, transform, fusion_level)
        self.dataset = self.dataset[:int(0.7 * len(self.dataset))]


class ValidationLoader(ImageLoader):
    def __init__(self, dataDir, split, transform, fusion_level):
        super().__init__(dataDir, split, transform, fusion_level)
        self.dataset = self.dataset[int(0.7 * len(self.dataset)):]