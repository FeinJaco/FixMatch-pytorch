import torch
from torchvision import datasets
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import torchvision.transforms as transforms

from dataset.cifar import DATASET_GETTERS

y_pred = []
y_true = []

path = './results/cifar10@250.5/model_best.pth.tar'
model = torch.load(path, map_location=torch.device("cpu"))

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
])
testset = datasets.CIFAR10(
        './data', train=False, transform=transform_val, download=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                        shuffle=False, num_workers=2)
print('start')
print(testloader)
# iterate over test data
for inputs, labels in testloader:
        print(len(y_pred))
        print('a')
        output = model(inputs) # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth

# constant for classes
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('output.png')
