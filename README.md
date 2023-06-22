# Image-Classification-Resnet34
## Introduction:
By using model Resnet-18 which was built by myself, I used Pytorch to classify image of 10 animal types.

Here is my pytorch implementation of the model described in the [RESNET paper](https://arxiv.org/abs/1512.03385). 

`Note:` Because this is the model I built, instead of this is trained by COCO dataset, I trained by dataset of 10 animal classes, which I present later. 

## Dataset:
Statistics of datasets I used for experiments. These datasets could be download from [link](https://www.kaggle.com/datasets/alessiocorrado99/animals10)

| Classes | Train samples | Test samples |
|:---------:|:---------------:|:--------------:|
|    butterfly |    1902  |    210   |
|    cat       |    1508  |    160   |
|    chicken   |    2790  |    308   |
|    cow       |    1684  |    182   |
|    dog       |    4373  |    490   |
|    elephant  |   1306   |    140   |
|    horse     |   2357   |    266   |
|    sheep     |   1638   |    182   |
|    spider    |   4345   |    476   |
|    squirre   |   1680   |    182   |

## Settings:
For optimizer and learning rate, I use:
- **SGD** optimizer with different learning rates (0.01 in most cases).

Additionally, in the my model, I will set up 100 epochs (using early stopping if after 5 epochs, if there is not greater score, it will stop train proccess) ,which is seen as a loop over `batch_size: 16`

## Training:

If you want to train a model with default parameters, you could run:

python train_animal.py 
If you want to adjust your preference parameters, here is some option you can choose:
| Parameters | Abbreviation | Default | Description |
|:---------:|:---------------:|:---------:|:---------:|
|    --batch-size |    -b  |    16                                  |Select suitable batch size|
|    --data-path  |    -p  |    '../../'                            |directory contains dataset|
|    --lr         |        |    1e-2                                |modify learning rate|
|    --epochs     |    -e  |    100                                 |modify epoch number|
|    --log-path   |    -l  |    tensorboard                         |[directory contains metrics visualization](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)|
|    --checkpoint |   -sc  |    tensorboard/animals/epoch_last.pt   |directory which saves the train model|

 For example: python train.py -p dataset_location --log-patch directory-name 
