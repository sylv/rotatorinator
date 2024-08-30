# rotatorinator

Fixes image orientations using ✨ machine learning ✨ by predicting the current orientation of an image (0, 90, 180, 270 degrees) and rotating it to the correct orientation.

## performance

Achieves >99.2% accuracy on the validation set. 
I forget which training set I used, but it was around 160k images.

Struggles with some images (in testing, mostly pictures of flowers/plants that are hard to discern, close ups of objects, things like that).
Does pretty well though, after training I ran it on ~4k scanned images that it had never seen and only a half dozen were obviously wrong.

## usage

> [!CAUTION] The infer.py script will overwrite the input images. It may destroy metadata, re-encode them and lose quality, or photoshop clown faces on them. Only run this on a copy of your images. I might fix that someday.

`python ./infer.py ./path/to/images/that/you/are/fine/destroying/*`

## todo

- [ ] Use something other than tensorflow for inference, it takes like 5 seconds to start
- [ ] Training-aware quantization for smaller and faster models

## disclaimer

I don't know python. I don't know machine learning. I don't know if there are catastrophic mistakes in the model structure or training script. This was a sidequest while digitizing photos. Use at your own risk and don't run it on anything important. 