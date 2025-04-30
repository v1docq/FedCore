from torch.nn import Dropout
from torch.nn import Identity
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Sequential
from torch.nn import Sigmoid
class ObjectDetector(Module):
    def __init__(self, baseModel, numClasses) -> None:
        super(ObjectDetector, self).__init__()

        # intialize base model and number of classes
        self.baseModel = baseModel
        self.numClasses = numClasses

        # build regressor head for outputting the bounding
        # box coordinates
        self.regressor = Sequential(
            Linear(baseModel.fc.in_features, 128),
            ReLU(),
            Linear(128, 64),
            ReLU(),
            Linear(64, 32),
            ReLU(),
            Linear(32, 4),
            Sigmoid(),
        )

        # build the classifier head that predicts the
        # class labels
        self.classifier = Sequential(
            Linear(baseModel.fc.in_features, 512),
            ReLU(),
            Dropout(),
            Linear(512, 512),
            ReLU(),
            Dropout(),
            Linear(512, 9),
        )

        # set classifier of our base model to produce
        # outputs from last convolutional block
        self.baseModel.fc = Identity()

    # we take the output of the base model and pass it through our heads
    def forward(self, x):
        # pass the inputs through the base model and then obtain
        # predictions from two different branches of the network
        features = self.baseModel(x)
        bboxes = self.regressor(features)
        classLogits = self.classifier(features)

        # return outputs as tuple
        return (bboxes, classLogits)
