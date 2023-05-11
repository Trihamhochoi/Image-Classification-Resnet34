import numpy as np
from torchvision.transforms import ToTensor, ToPILImage, Compose, Resize, Normalize
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Animal_class import AnimalDataset
from cnn_model import AnimalSimpleConvNet
from sklearn.metrics import confusion_matrix, accuracy_score
from new_conf_matrix import createConfusionMatrix
from con_matrix import plot_confusion_matrix
from Ex7_resnet_model.resnet_model import ResNet
import os
import shutil
from tqdm.autonotebook import tqdm
import argparse
import matplotlib as plt

def get_args():
    parser = argparse.ArgumentParser(description="Train a CNN model")
    parser.add_argument("--batch-size", '-b', type=int, default=32)
    parser.add_argument("--data-path", '-p', type=str, default='../')
    parser.add_argument("--lr", type=int, default=1e-3)
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument("--log-path", '-l', type=str, default='tensorboard')
    parser.add_argument("--save-path", '-sp', type=str, default='tensorboard/animals')
    parser.add_argument("--checkpoint", '-sc', type=str, default=None) #'tensorboard/animals/epoch_last.pt'

    args = parser.parse_args()
    return args


# tensorboard --logdir demo/session_8/tensorboard/
args = get_args()
batch_size = args.batch_size
num_epochs = args.epochs

if __name__ == '__main__':
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])
    # Create Train set and Test set
    train_set = AnimalDataset(root=args.data_path, train=True, transform=transform)
    test_set = AnimalDataset(root=args.data_path, train=False, transform=transform)

    # Create Data Loader
    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=4,
        drop_last=True,
        shuffle=True,
        pin_memory=False
    )
    test_dataloader = DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        num_workers=4,
        drop_last=False,
        shuffle=False,
        pin_memory=False
    )

    # SET UP MODEL, FOLDER AND PARAMETER BEFORE TRAINING AND TESTING PROCESS
    # Set device is CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    #model = AnimalSimpleConvNet(num_classes=10).to(device)
    model = ResNet(num_classes=10).to(device)

    # Create Gradient Decent
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # set up checkpoint:
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    # create directory containing the model
    # if directory existed, remove that directory, create new one
    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.mkdir(args.log_path)

    if os.path.isdir(args.save_path):
        shutil.rmtree(args.save_path)
    os.mkdir(args.save_path)

    # Create Tensor board
    writer = SummaryWriter(log_dir=args.log_path)

    # Get the best loss:
    best_loss = 10000
    best_epoch = 0
    # START TRAIN AND EVALUATE OF MODEL
    for epoch in range(start_epoch, args.epochs):
        # TRAINING STEP
        model.train()
        train_loss = []

        # Create process bar when running each epoch
        process_bar = tqdm(train_dataloader, colour='blue')
        for i, (images, labels) in enumerate(process_bar):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss_value = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()  # no need to save in the optimize list
            loss_value.backward()
            train_loss.append(loss_value.item())

            # OPTIMIZATION
            optimizer.step()  # update parameter

            process_bar.set_description(
                "Epoch {}. Iteration {}/{} Loss {}. lr: {}".format(epoch + 1,
                                                                   i + 1,
                                                                   len(train_dataloader),
                                                                   np.mean(train_loss),
                                                                   optimizer.param_groups[-1]['lr']
                                                                   )
            )

            # Create metric
            writer.add_scalar("Train/Loss", np.mean(train_loss), i + epoch * len(train_dataloader))

        # VALIDATION STEP
        all_predictions = []
        all_labels = []
        test_loss = []

        # set model in the evaluation mode
        model.eval()

        # Stop the optimal process
        with torch.inference_mode():

            # START TO EVALUATE THE TEST DATALOADER
            for i, (images, labels) in enumerate(test_dataloader):
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)

                # Calculate the loss value
                loss_value = criterion(outputs, labels)
                test_loss.append(loss_value.item())

                # Get the prediction
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.tolist())
                all_labels.extend(labels.tolist())

            # Calculate the accuracy score
            acc = accuracy_score(all_labels, all_predictions)
            print("=== epoch {}. Accuracy {} ===".format(epoch + 1, acc))

            # Create metrics to evaluate the model
            writer.add_scalar(tag="EVAL/LOSS",
                              scalar_value=np.mean(test_loss),
                              global_step=epoch)

            writer.add_scalar(tag="EVAL/ACC",
                              scalar_value=acc,
                              global_step=epoch)

            # Create confusion matrix
            conf_matrix = confusion_matrix(all_labels, all_predictions)
            fig = createConfusionMatrix(conf_matrix=conf_matrix,
                                        classes_=test_set.categories)

            writer.add_figure(tag="Confusion matrix",figure=fig,global_step=epoch)

            # plot_confusion_matrix(writer,
            #                       conf_matrix,
            #                       test_set.categories,  # class name
            #                       epoch)

        # Create checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        args.checkpoint = checkpoint

        # Save model in the last epoch we run
        torch.save(checkpoint, os.path.join(args.save_path, 'epoch_last.pt'))
        if np.mean(test_loss) < best_loss:
            best_loss = np.mean(test_loss)
            best_epoch = epoch +1
            torch.save(checkpoint, os.path.join(args.save_path, 'epoch_best.pt'))

        if epoch - best_epoch == 5:  # Early Stopping
            exit(0)
