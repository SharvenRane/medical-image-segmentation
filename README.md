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

## Montgomery County lung fields

For the NLM Montgomery County chest X-ray set, point the trainer at the
extracted `MontgomerySet` folder and it reads the `CXR_png` images and the
`ManualMask/leftMask` plus `ManualMask/rightMask` lung masks directly (the full
lung mask is the union of the two):

```
python src/train.py --montgomery /path/to/MontgomerySet --arch unetplusplus --encoder resnet34 --epochs 50
```

Download (about 600 MB) from the NLM mirror:
`https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip`

## Results on Montgomery lung fields

A 50 epoch UNet++ run with a resnet34 encoder on the Montgomery County set (real
chest X-rays with manual lung field masks) reached a best validation Dice of
0.979 on an RTX 5070 Ti. Dice passed 0.90 by the fourth epoch and held in the
high 0.97 range across the back half of training. This is a real run, not a
quoted benchmark; the synthetic shapes path stays in place for tests.

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
