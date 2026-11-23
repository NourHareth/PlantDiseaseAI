# Defence notes: questions an examiner is likely to ask

Short, honest answers to the questions this project invites. Read the code next to each answer;
you should be able to explain every line it points to.

---

## The results in one minute

1. On lab photos, fine-tuning is what matters (+10 points over a frozen backbone). The backbone and
   resolution do not matter significantly; pre-training adds 1.7 points over training from scratch.
2. On field photos every lab-trained model scores 21–29%, barely above always guessing the most
   common class (20%). Lab accuracy does not predict field accuracy.
3. The models send about 90% of field photos to three classes and almost never say "healthy",
   while reporting 61–89% confidence. Calibration fitted on lab data does not transfer.
4. Strong augmentation is the only lab-only fix that significantly helps in the field (+5 points,
   p = 0.013). Background replacement adds nothing on top of it.
5. The Mahalanobis detector on frozen ImageNet features catches non-leaf images almost perfectly,
   and it flags 95% of field photos as unfamiliar, so the app warns users when it should.
6. Dynamic-range TFLite: 3.5× smaller, same accuracy. Full int8: fastest, but 4–7 points worse.

**If asked "so is your model good?":** in the lab, yes (95–97%). In a real field, no, and the
project measures exactly how and why, and the app tells the user when a photo is outside what the
model knows. That honesty is the contribution.

---

## Data

**Why not just use a random train/test split?**
Some PlantVillage leaves were photographed several times (`GHLB2 Leaf 121`, `121.1`, … `121.4`).
With a random split, one photo of a leaf can land in training and another in test, and the model is
then partly tested on what it has already seen. `scripts/prepare_data.py` groups images by leaf ID,
identical bytes (MD5) and identical appearance (dHash), and assigns whole groups to one split.
The audit in `data/splits/audit.json` reports how many images this affects.

**What is a dHash?** Shrink the image to 9×8 grey pixels and record, for each row, whether each pixel
is brighter than its right neighbour: 64 bits. Re-encoded or resized copies of a picture keep
the same bits, so equal hashes find visual duplicates that MD5 misses.

**Why is the split stored in a CSV instead of being re-created each run?**
So every experiment uses exactly the same images (fair comparison), and anyone can reproduce the
numbers. Re-splitting with a different library version or file order could silently change the test set.

**Why evaluate on PlantDoc?**
PlantVillage is lab photos: one leaf, plain background, controlled light. Accuracy there says little
about a farmer's phone photo. PlantDoc contains real-world images, so it measures *domain
generalisation*. It is used only for testing (except the explicitly labelled `*_mixed` run).

**Was the original `class_names.json` in the right order?** It looked suspicious: it is sorted
case-insensitively (`healthy` third), while Keras numbers class folders case-sensitively (`healthy`
last). `scripts/evaluate.py` scored all 18,160 PlantVillage images under both interpretations:
89.74% with the file's order, 16.44% with Keras' order. So the file is correct. It was probably
written in the order of the folders as the original training code listed them.

**Why does the field evaluation use all of PlantDoc instead of its test split?** After removing
duplicates, PlantDoc's tomato test split has only 64 images, which gives an accuracy interval of
about ±12 points. Models that never trained on PlantDoc can be scored on all 741 images without
any leakage. Only the mixed-domain run, which trains on PlantDoc's training split, is limited to the 64.

**Why are some classes missing from PlantDoc results?** PlantDoc has no *Target spot* class, and its
test split has no spider-mite images. Accuracy is computed over the classes that are present; the
model can still (wrongly) predict a missing class.

## Model and training

**Why MobileNetV2?** It is small (2.3 M parameters, ~9 MB) and fast, built for phones, and its ImageNet
features transfer well. EfficientNetB0 and ResNet50V2 are compared to show the trade-off (Section 5 of the report).

**What does the `Rescaling(2, -1)` layer do?** The app feeds images scaled to [0, 1]. MobileNetV2 was
pre-trained on [-1, 1]. `x * 2 - 1` maps one to the other *inside* the model, so preprocessing can
never drift between training and deployment. EfficientNet expects [0, 255], so its factor is 255.

**Frozen vs fine-tuned vs from scratch, in one sentence each.**
Frozen trains only the new classifier on fixed ImageNet features. Fine-tuning then also adapts the top
layers of the backbone with a 100× smaller learning rate. Scratch learns everything from random weights.

**Why keep BatchNorm frozen while fine-tuning?** BatchNorm layers keep running statistics of their
inputs. Updating them with small, differently-distributed batches would wreck the pretrained
features in the first steps. Calling the backbone with `training=False` and keeping BN layers
non-trainable is the approach recommended in the Keras transfer-learning guide.
(`plantdisease/backbones.py`, `unfreeze_top`)

**But then why does the scratch run train BatchNorm?** With random weights there are no useful
statistics to protect; the network must learn them (`train_batchnorm=True`).

**How was the fine-tuning learning rate chosen?** The first attempt used 1e-5 for 8 epochs, the
conservative value from the Keras guide. Validation loss was still falling at the last epoch, which
means under-training rather than a ceiling, so it was raised to 1e-4 for up to 12 epochs. The
decision rested on the validation curve. The first run's test score was printed as well, but no
setting was chosen by comparing test scores; this was the only re-tuning done.

**The first from-scratch run predicted one class for everything. Why?** BatchNorm normalises with
the current batch's statistics during training, and with a moving average of them at evaluation
time. Keras' MobileNetV2 updates that average with momentum 0.999, roughly the last 1,000 steps.
From random weights the network changes much faster than that, so at evaluation time the averages
describe an older network, and every image ended up as *Bacterial spot* (11.7%, that class's
share). Proof: re-estimating the statistics on 1,280 training images, without touching a weight,
lifted validation accuracy from 11.8% to 44%. Fix: momentum 0.9 for scratch runs
(`SCRATCH_BN_MOMENTUM`). Fine-tuned models never hit this because their BatchNorm is frozen.

**Why early stopping on validation loss, with restore_best_weights?** To stop when the model starts
over-fitting and keep the best epoch's weights rather than the last.

**What is background replacement and why should it help?**
PlantVillage backgrounds are uniform lab surfaces, and a network can learn shortcuts from them. Using the
dataset's segmented images as masks, the leaf is pasted onto random synthetic backgrounds during
training (`plantdisease/pipeline.py`), so the background carries no information. This is *domain
randomisation*.

**Did it work?** Not beyond strong augmentation: 27.1% ± 2.0 field accuracy over three seeds vs
27.0% for strong augmentation alone, and it costs lab accuracy. The likely reasons, all in the
report:
- The fine-tuned models already looked at the leaf (Grad-CAM lift 1.40 either way).
- The PlantVillage masks cut out dark lesions, so replacement erased some disease evidence.
- Colour-noise backgrounds do not look like real fields.

A negative result reported with its reasons is still a result.

## Evaluation

**What does the 95% confidence interval mean?** The test set is one sample of possible images.
Bootstrapping resamples the test predictions 2,000 times; the middle 95% of the resulting accuracies
shows how much the score could move by chance. Overlapping intervals mean "not clearly different".

**Why three seeds?** Training is random (initialisation, shuffling, augmentation). The spread across
seeds shows whether a difference between configurations is bigger than run-to-run noise.

**How do you know one setting really beats another?** Two different sources of chance, two tools.
Seeds show how much *training* randomness moves a score. The exact **McNemar test**
(`scripts/compare_runs.py`) asks whether the gap could come from *which test images* happened to be
drawn. Both models are scored on the same images, so the images both get right or both get wrong
say nothing about the difference; only the disagreements count. If the models were equally good,
each disagreement would favour either side with probability ½, so the count is Binomial(n, ½).
This is far sharper than checking whether two confidence intervals overlap, because it uses the
pairing.

**Why the Holm correction?** About ten comparisons are tested at once. At p < 0.05 each, the chance
of at least one false "significant" result is far above 5%. Holm multiplies the smallest p-value by
the number of tests, the next by one fewer, and so on, which keeps that overall chance at 5% or
below.

**Macro-F1 vs accuracy?** Accuracy is dominated by large classes. Macro-F1 averages the per-class F1,
so every disease counts equally.

## Calibration

**What is calibration?** A model is calibrated if, among predictions made with 80% confidence, about 80%
are correct. ECE (expected calibration error) is the weighted average gap between confidence and
accuracy over confidence bins.

**What is temperature scaling, and why doesn't it change accuracy?** Divide the logits by one number
T before the softmax. T > 1 flattens the probabilities. Dividing by a positive constant never
changes which logit is largest, so the predicted class and the accuracy stay the same; only
the confidence changes. T is fitted on the validation set by minimising negative log-likelihood.

**How do you get logits from a model whose last layer is a softmax?** `log(p)` equals the logits up
to a constant per image, and the softmax ignores constants
(`plantdisease/calibration.py`, `apply_temperature`).

## Out-of-distribution detection

**Why can't the model just say "not a leaf"?** A softmax classifier must distribute 100% over its
classes. A black square still gets "Late blight 99.5%". OOD detection adds a separate score that
measures how familiar an input looks.

**The three scores:**
- *Max softmax probability*: the baseline; often fooled, because networks are confidently wrong.
- *Energy*: `logsumexp(logits)`, using the raw logits before the softmax squashes them.
- *Mahalanobis*: distance of the 1,280-dimensional penultimate features to the nearest class centre,
  measured with the shared covariance of the training features. Far from every class means unfamiliar.

**What is AUROC?** The probability that a random real test photo scores higher than a random
non-leaf image. 0.5 = useless, 1.0 = perfect separation. It does not depend on a threshold.

**How is the threshold chosen in the app?** So that 95% of real validation photos are accepted
(`calibrate_threshold`). The remaining 5% false alarms are the price of catching unfamiliar images.

## Explainability

**How does Grad-CAM work?** Take the gradient of a class score with respect to a convolutional feature
map, average it per channel to get channel weights, form the weighted sum of the channels, keep
the positive part, and upsample. Bright regions increased that class's score.

**Why use the logit instead of the softmax output?** At 99.9% confidence the softmax is saturated
and its gradients become tiny and distorted. The logit's gradient is not squashed.

**Why the 8×8 layer and not the last one?** At 128 px input MobileNetV2's last block is 4×4, only 16
cells for a whole leaf. The deepest 8×8 layer keeps class-specific meaning with 4× the resolution.

**What is "lift"?** Grad-CAM attention inside the leaf mask divided by the leaf's share of the image.
Lift 1.0 is what you would get by spreading attention uniformly; lift > 1 means the model
concentrates on the leaf. It turns "the heatmap looks reasonable" into a number.

## Deployment

**Dynamic-range vs int8 quantisation?** Dynamic range stores weights as 8-bit integers (about 4×
smaller) but computes in float. Full int8 also quantises activations, using ~200 calibration
images to choose the ranges; it is fastest on phone NPUs, at a possible small accuracy cost.

**Why are augmentation and dropout removed before export?** They only act during training. Removing
them gives a smaller, cleaner inference graph (`scripts/export_tflite.py`, `inference_model`).

## Software engineering

**Why a package plus thin scripts instead of a notebook?** Notebooks hide state and are hard to test.
Here the same `plantdisease` code is used by the app, the scripts and the tests, so the numbers in
the report come from exactly the code the app runs.

**How do you know it works?** The pytest suite covers unit maths (AUROC, ECE, temperature fitting),
data handling (grouping, manifests, background replacement), every backbone, the full training and
evaluation chain on tiny data, the TFLite export and the Streamlit pages. CI runs it on every push.

**Why is the original model still deployed?** It was a deliberate decision to keep the app's model
unchanged; the experiments are reported separately in the Experiments page and the report.
Any run can be deployed by copying its files into `models/`.
