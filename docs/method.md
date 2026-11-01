# Method

## Conventions

- Points are `(x, y)`. Arrays are indexed `[row, col]`, so a point `(x, y)` is `array[y, x]`.
- Shapes are `(height, width)`.
- Names carry their unit: `_px`, `_mm`, `_cm`.
- Images are BGR, as loaded by OpenCV.

## Inputs

- A colour image `H x W x 3`.
- A depth map `H x W` **aligned to the colour image** (same size, same viewpoint), in
  `depth_unit_mm` millimetres per unit. A 16-bit depth image in millimetres uses `1`; depth
  stored as floating-point metres uses `1000`.
- A segmentation of the box: from an ONNX model, or a binary mask.

## Geometry

The camera looks straight down at a flat base. With the pinhole model, an object at distance
`d` that spans `s` pixels has a real size of `s * d / f`, where `f` is the focal length in
pixels.

1. **Footprint.** `cv2.minAreaRect` on the mask pixels gives an oriented rectangle. The longer
   side is the length, the shorter side is the width.
2. **Surface distance `d`.** Take the depth samples **under the mask** and convert them to
   centimetres. Discard non-finite and non-positive values. Keep the values between the low
   and high percentile (default 20 and 80), average them, and multiply by `distance_factor`.
   The percentile band makes the result robust to mask bleeding onto the floor, to holes in
   the depth map and to edge noise.
3. **Dimensions.**

   ```
   length = length_px * d / f
   width  = width_px  * d / f
   height = camera_height - base_height - d
   ```

4. **Volume** is `length * width * height`.

When only one or two valid samples exist, the percentile band can be empty; the median is
used instead.

## Failures

The tool raises an error instead of returning a doubtful number:

| Situation | Error |
|---|---|
| The mask is empty, or the model detects nothing above the threshold | no object |
| Every depth sample under the mask is zero or invalid | no valid depth |
| `camera_height - base_height - d <= 0` | non-positive height (usually a calibration problem) |

## Segmentation model contract

The model is an ONNX instance-segmentation network with the YOLOv8-seg output layout.

- **Input** `(1, 3, H, W)`, float32, RGB, scaled to `[0, 1]`. The image is resized to `(W, H)`
  with no letterboxing. If the input axes are dynamic, `640 x 640` is used.
- **`output0`** `(1, 4 + C + M, N)`: for each of `N` candidates, the box centre `(cx, cy)` and
  size `(w, h)` in input pixels, `C` class scores, and `M` mask coefficients.
- **`output1`** `(1, M, mh, mw)`: mask prototypes.

Decoding:

1. Score of a candidate = its best class score. Keep scores `>= --confidence` (default 0.5).
2. Convert boxes to corners and apply greedy non-maximum suppression (IoU 0.5, class-agnostic).
3. For each survivor, mask = `sigmoid(coefficients @ prototypes)`, resized to the original
   image, thresholded at 0.5 and cropped to the box.
4. Among the survivors, the one whose box centre is nearest the reference point (the image
   centre unless `reference_point` is set) is measured.

Execution providers default to whatever ONNX Runtime reports as available.

This decoding is covered by tests on synthetic tensors only. Before relying on a model, check
that its outputs have the shapes above.

## Calibration procedure

1. **`camera_height_cm`**: measure from the camera's optical centre to the surface the box
   rests on, with a tape measure.
2. **`base_height_cm`**: height of a pallet or platform between that surface and the box;
   `0` if the box rests directly on it.
3. **`focal_length_px`**: use the `fx` intrinsic of the colour stream. With a RealSense camera,
   `depth-volumetry capture` prints it. Intrinsics depend on the resolution, so read them at the
   resolution you measure at.
4. **`depth_unit_mm`**: for RealSense, `capture` prints it as well (typically `1`).
5. **Check** with a box of known size: measure it and compare. If the length and width are
   off by a consistent ratio, adjust `distance_factor` or re-check `focal_length_px`; if the
   height is off, re-check `camera_height_cm`.

## Assumptions and limits

- Flat base, camera pointing straight down, box top parallel to the base.
- The depth map and the colour image are aligned.
- One box per measurement.
- Accuracy is bounded by calibration and depth-sensor noise.
