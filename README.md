# Medical Image Segmentation

Binary segmentation for medical images with UNet++ and DeepLabV3+ from
segmentation-models-pytorch, a Dice plus BCE loss, and Dice as the metric. This
is the setup you reach for on masks like lung fields, polyps or lesions.

## Your own data

Put images and masks in two folders with matching filenames and point the
trainer at them:

```
pip install -r requirements.txt
python src/train.py --images data/images --masks data/masks --arch unetplusplus --encoder resnet34 --epochs 30
```

Any binary medical mask set in that layout works, chest X-ray lung masks for
instance.

## What is inside

`model.py` builds UNet++, DeepLabV3+ or plain UNet over a choice of encoders.
`losses.py` has the Dice plus BCE loss and the Dice metric. `dataset.py` reads
paired image and mask folders, and it can also generate synthetic shapes so the
pipeline is testable with nothing to download. `train.py` is the loop with a
Dice validation and best checkpoint saving.

## Tests

```
pytest tests/ -q
```

The tests build both architectures, confirm Dice is 1.0 on a perfect mask and
near zero on an inverted one, and check that the loss actually goes down over a
few real steps on the synthetic shapes.
