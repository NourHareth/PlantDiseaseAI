**What 14 training runs found** (details in `reports/REPORT.md`):

- **Lab accuracy does not carry over to the field.** Models score 85–97% on lab photos but 21–29% on
  real field photos, close to always guessing the most common field class (20%). About 90% of field
  photos land in three classes, and healthy field leaves are almost never recognised.
- **Strong augmentation is the one lab-only change that helped in the field:** +5.0 points
  (p = 0.013, paired test). Background replacement, other backbones and larger images did not.
- **Confidence is only trustworthy on lab-like photos.** Lab calibration error is ≤ 0.02, yet in
  the field the models report 61–89% confidence. The app's unfamiliar-photo detector flags 95% of
  field photos, which is why the Diagnose page warns.
