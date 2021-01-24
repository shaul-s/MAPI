import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

import albumentations as albu

import torch
import numpy as np
import segmentation_models_pytorch as smp

DATA_DIR = r'C:\Users\Saul\PycharmProjects\just_for_fun\MapiX\new_moshe_data\coco\dataset'

x_train_dir = os.path.join(DATA_DIR, r'train\stamps')
y_train_dir = os.path.join(DATA_DIR, r'train\annotations')

x_valid_dir = os.path.join(DATA_DIR, r'eval\stamps')
y_valid_dir = os.path.join(DATA_DIR, r'eval\annotations')

x_test_dir = os.path.join(DATA_DIR, r'test\stamps')
y_test_dir = os.path.join(DATA_DIR, r'test\annotations')

TRAINING = 1


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ['255']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.img_ids = os.listdir(images_dir)
        self.msk_ids = os.listdir(masks_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.img_ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.msk_ids]

        # convert str names to class values on masks
        # self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.class_values = [255]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.img_ids)


# Lets look at data we have

# dataset = Dataset(x_train_dir, y_train_dir, classes=['building'])
#
# image, mask = dataset[6]  # get some sample
# visualize(
#     image=image,
#     building_mask=mask.squeeze(),
# )

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(320, 320)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


#### Visualize resulted augmented images and masks

# augmented_dataset = Dataset(
#     x_train_dir,
#     y_train_dir,
#     augmentation=get_training_augmentation(),
#     classes=['building'],
# )
#
# # same image with different random transforms
# for i in range(3):
#     image, mask = augmented_dataset[1]
#     visualize(image=image, mask=mask.squeeze(-1))

ENCODER = 'resnet101'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['255']
ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create segmentation model with pretrained encoder
model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

valid_dataset = Dataset(
    x_valid_dir,
    y_valid_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index
loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.0001),
])

# create epoch runners
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

if __name__ == '__main__':
    # train model for 40 epochs
    if TRAINING:
        max_score = 0

        optimizer.param_groups[0]['lr'] = 0.001

        for i in range(0, 40):

            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader)
            # valid_logs = valid_epoch.run(valid_loader)
            torch.save(model, r'./data/smp_1st_tst/best_model_trees.pth')
            print('Model saved!')

            # do something (save model, change lr, etc.)
            # if max_score < valid_logs['iou_score']:
            #     max_score = valid_logs['iou_score']
            #     torch.save(model, r'./data/smp_1st_tst/best_model_trees.pth')
            #     print('Model saved!')

            # if i == 25:
            #     optimizer.param_groups[0]['lr'] = 1e-5
            #     print('Decrease decoder learning rate to 1e-5!')

    # load best saved checkpoint
    best_model = torch.load(r'./data/smp_1st_tst/best_model_trees.pth')

    # create test dataset
    # test_dataset = Dataset(
    #     x_test_dir,
    #     y_test_dir,
    #     augmentation=get_validation_augmentation(),
    #     preprocessing=get_preprocessing(preprocessing_fn),
    #     classes=CLASSES,
    # )
    #
    # test_dataloader = DataLoader(test_dataset)
    #
    # # evaluate model on test set
    # test_epoch = smp.utils.train.ValidEpoch(
    #     model=best_model,
    #     loss=loss,
    #     metrics=metrics,
    #     device=DEVICE,
    # )
    #
    # logs = test_epoch.run(test_dataloader)
    #
    # test dataset without transformations for image visualization
    # test_dataset_vis = Dataset(
    #     x_test_dir, y_test_dir,
    #     classes=CLASSES,
    # )
    #
    test_dataset = Dataset(
        x_test_dir,
        y_test_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    test_dataset_vis = Dataset(
        x_test_dir, y_test_dir,
        classes=CLASSES,
    )

    # # load best saved checkpoint
    # best_model = torch.load(r'./data/smp_1st_tst/best_model.pth')
    # image_vis = cv2.imread(r'C:\Users\Saul\PycharmProjects\just_for_fun\MapiX\data\smp_1st_tst\test\123_X3-Y17__896_-1386.tif')
    # image_vis = cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB).astype('uint8')
    #

    # x_tensor = torch.from_numpy(test_dataset[0][0]).to(DEVICE).unsqueeze(0)
    # pr_mask = best_model.predict(x_tensor)
    # pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    # #
    # visualize(
    #     image=test_dataset[0][0],
    #     predicted_mask=pr_mask
    # )

    for i in range(17):
        # n = np.random.choice(len(test_dataset))
        n = i + 1

        image_vis = test_dataset_vis[n][0].astype('uint8')
        image, gt_mask = test_dataset[n]

        # gt_mask = gt_mask.squeeze()

        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        visualize(
            image=image_vis,
            predicted_mask=pr_mask
        )


        # im = Image.fromarray(image_vis)
        # ma = Image.fromarray(pr_mask)
        # im.save("stamp{}.tiff".format(i))
        # ma.save("pred{}.tiff".format(i))
