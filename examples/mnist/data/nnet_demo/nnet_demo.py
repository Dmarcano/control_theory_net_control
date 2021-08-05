######################################################################
# nnet_demo.py - Neural Network demo
# Writen for ENGR 027 - Computer Vision
# Matt Zucker 2020-2021
######################################################################

import struct
import cv2
import numpy as np

IMAGE_SIZE = 28
INSET_SIZE = 20
INSET_MARGIN = (IMAGE_SIZE - INSET_SIZE)//2

NUM_INPUT = IMAGE_SIZE*IMAGE_SIZE
NUM_HIDDEN = 300
NUM_OUTPUT = 10

PIXEL_SCALE = 16

PEN_SIZE = 28

WINDOW_SIZE = IMAGE_SIZE*PIXEL_SCALE
WINDOW_INSET_SIZE = INSET_SIZE*PIXEL_SCALE
WINDOW_INSET_MARGIN = INSET_MARGIN*PIXEL_SCALE

SCALE_MIN = 0.5 / PIXEL_SCALE
SCALE_MAX = 2.0 / PIXEL_SCALE

WINDOW_TITLE = 'MNIST Demo'

######################################################################
# preprocess image array of dtype uint8 and shape (n, 28, 28)
# into "sphered" array of dtype float32 and shape (n, 784)
# https://en.wikipedia.org/wiki/Whitening_transformation

def preprocess(imgs):
    
    assert len(imgs.shape) == 3
    assert imgs.shape[1:] == (IMAGE_SIZE, IMAGE_SIZE)

    v = imgs.reshape(-1, NUM_INPUT).astype(np.float32)

    v -= v.mean(axis=1).reshape(-1, 1) # subtract row-wise mean

    s = v.std(axis=1).reshape(-1, 1)
    v /= np.where(s, s, 1)  # divide off row-wise stddev but avoid divide by zero

    return v

######################################################################
# predict classes for batch of data

def predict(weights, x):

    # x.shape == (num_samples, num_features)

    for w in weights:
        
        # w.shape == (num_layer_inputs+1, num_layer_outputs)
        
        # all columns but last correspond to weight terms
        layer_weights = w[:-1]
        
        # last column corresponds to bias terms
        layer_bias = w[-1]

        # matrix multiplication plus bias
        xw = np.dot(x, layer_weights) + layer_bias

        # pass thru nonlinear tanh activation function
        x = np.tanh(xw)

    # return final classification = max over all class outputs
    # per-sample -- shape is just (num_samples,)
    return x.argmax(axis=1)

######################################################################
# downscale the canvas to a 28x28 image using the mnist
# standardization techniques:
#
#   "The original black and white (bilevel) images from NIST were size
#    normalized to fit in a 20x20 pixel box while preserving their
#    aspect ratio. The resulting images contain grey levels as a
#    result of the anti-aliasing technique used by the normalization
#    algorithm. the images were centered in a 28x28 image by computing
#    the center of mass of the pixels, and translating the image so as
#    to position this point at the center of the 28x28 field."
#
# quoted from http://yann.lecun.com/exdb/mnist/

def downscale(canvas):

    if not np.any(canvas):
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

    # dimensions of bounding box
    dims = []

    # centroid of bounding box
    centroid = []

    # indices of rows/cols
    idx = np.arange(WINDOW_SIZE)

    # for each axis
    for axis in range(2):

        # get the mean intensity of the image along this axis
        mean_across_axis = canvas.mean(axis=axis)

        # find nonzero rows/cols
        nz = idx[mean_across_axis != 0]

        # get dimension along this axis
        dims.append(nz.max() - nz.min())

        # compute centroid along this axis
        centroid.append( np.dot(idx, mean_across_axis) / mean_across_axis.sum() )

    # find maximum dimension
    dmax = max(dims)

    # get scale factor to resize this to INSET_SIZE
    scale = INSET_SIZE / dmax

    # force to min/max scale bounds
    scale = np.clip(scale, SCALE_MIN, SCALE_MAX)

    # we will do scaling/translating in two steps to work around
    # limitations of cv2.warpAffine -- namely that it does not support
    # cv2.INTER_AREA :(
    k = 4

    # first translate by negative centroid
    cx, cy = centroid
    T0 = np.matrix([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])

    # next scale by computed scale factor (times a constant)
    S = np.matrix([[k*scale, 0, 0], [0, k*scale, 0], [0, 0, 1]])

    # next translate by positive small image center
    mx, my = IMAGE_SIZE//2, IMAGE_SIZE//2
    T1 = np.matrix([[1, 0, k*mx], [0, 1, k*my], [0, 0, 1]])

    # compose transformations
    M = np.array(T1 * S * T0)[:2]

    # step 1: affine transform of image (uses cv2.INTER_LINEAR sadly)
    warp_big = cv2.warpAffine(canvas, M, (k*IMAGE_SIZE, k*IMAGE_SIZE))

    # step 2: smoother downscale
    warp = cv2.resize(warp_big, (IMAGE_SIZE, IMAGE_SIZE),
                      interpolation=cv2.INTER_AREA)

    return warp


######################################################################
# class for running this demo

class NNetDemo(object):

    # constructor
    def __init__(self):

        # load weights
        with np.load('mnist_weights_quantized.npz') as npzfile:
            self.weights = []
            for key in npzfile.files:
                w = npzfile[key]
                # weights are stored in the file as signed 8-bit integers
                # where +/- 127 corresponds to +/- 0.5
                w = w.astype(np.float32) * 0.5 / 127
                self.weights.append(w)

        # load test data
        with np.load('sample_data.npz') as npzfile:
            labels = npzfile['labels']
            images = npzfile['images']

        # shuffle so user sees different thing each program run
        idx = np.arange(len(labels))
        np.random.shuffle(idx)
        self.sample_labels = labels[idx]
        self.sample_images = images[idx]

        # preprocess into (n-by-784) array of floats
        self.sample_vectors = preprocess(self.sample_images)

        # make predictions on test set and print error rate
        self.sample_predictions = predict(self.weights, self.sample_vectors)
        is_correct = (self.sample_predictions != self.sample_labels)
        error_rate = is_correct.mean()
        print('error rate on sample dataset: {:.2f}%'.format(error_rate*100))

        # initialize some GUI variables
        self.interactive = False
        self.show_label = False
        self.prev_mouse_pos = None
        self.canvas = np.zeros((WINDOW_SIZE, WINDOW_SIZE), dtype=np.uint8)

        self.set_cur_idx(0)

        # set up OpenCV gui
        wflags = cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_AUTOSIZE
        cv2.namedWindow(WINDOW_TITLE, wflags)
        cv2.setMouseCallback(WINDOW_TITLE, self.mouse_event, None)

    # update current index in test set and trigger redraw
    def set_cur_idx(self, idx):

        self.cur_idx = idx % len(self.sample_labels)
        self.window = None

    # called when mouse pressed/moved/released
    def mouse_event(self, event, x, y, flags, param):

        # ignore if not in interactive mode
        if not self.interactive:
            return

        # are we dragging
        dragging = ( event == cv2.EVENT_LBUTTONDOWN or
                     flags & cv2.EVENT_FLAG_LBUTTON )

        # capture coords from right side of window only
        # but map them to the left
        x = max(x - WINDOW_SIZE, 0)

        mouse_pos = (x, y)

        if event == cv2.EVENT_LBUTTONUP:
            
            # handle mouse up
            self.prev_mouse_pos = None
            
        elif dragging:
            
            # handle press/move
            if self.prev_mouse_pos is None:
                # mouse press
                cv2.circle(self.canvas, mouse_pos, PEN_SIZE//2,
                           (255, 255, 255), -1, cv2.LINE_AA)
            else:
                # mouse move
                cv2.line(self.canvas, self.prev_mouse_pos, mouse_pos,
                         (255, 255, 255), PEN_SIZE, cv2.LINE_AA)
                
            # update previous mouse pos
            self.prev_mouse_pos = mouse_pos

            # trigger redraw
            self.window = None

    # redraw the window and make a prediction if necessary
    def redraw_window(self):

        if self.interactive:
            # if in interactive mode, downscale the canvas to make the input image
            self.input_image = downscale(self.canvas)
            # then preprocess into 1x784 vector
            v = preprocess(self.input_image.reshape(1, IMAGE_SIZE, IMAGE_SIZE))
            # then predict on this "batch" of size 1
            p = predict(self.weights, v)
            # and set the current prediction to the first element
            self.prediction = p[0]
        else:
            # not in interactive mode so just grab input image and
            # prediction from loaded/precomputed data.
            self.input_image = self.sample_images[self.cur_idx]
            self.prediction = self.sample_predictions[self.cur_idx]

        # upscale display
        display = cv2.resize(self.input_image, (WINDOW_SIZE, WINDOW_SIZE),
                             interpolation=cv2.INTER_NEAREST)

        # place it side-by-side with canvas if necessary
        if self.interactive:
            # contrast-reduced image
            tmp = (0.9 * self.canvas.astype(np.float32) + 0.1*255).astype(np.uint8)
            # stack horizontally
            display = np.hstack((display, tmp))

        # invert grayscale and convert to color image
        self.window = cv2.cvtColor(255 - display, cv2.COLOR_GRAY2RGB)

        # draw label
        if self.show_label and np.any(self.input_image):
            cv2.putText(self.window, str(self.prediction),
                        (8, WINDOW_SIZE-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 0, 255), 2, cv2.LINE_AA)

        # draw help text
        y = 20
        lines =  ['Press SPACE to advance/reset,',
                  'L to toggle label,',
                  'I to toggle interactive mode']

        for line in lines:
            cv2.putText(self.window, line,
                        (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 0, 255), 1, cv2.LINE_AA)
            y += 15

        # box annotations
        if self.interactive:
            
            s = WINDOW_SIZE
            r0 = WINDOW_INSET_MARGIN
            r1 = r0 + WINDOW_INSET_SIZE
            
            cv2.rectangle(self.window, (s+r0, r0), (s+r1, r1), (255, 0, 255), 1)

            cv2.putText(self.window, 'Draw a digit inside the box!',
                        (s+8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 0, 255), 1, cv2.LINE_AA)
            

    # clear the canvas and trigger a redraw
    def reset_interactive(self):
        self.canvas[:] = 0
        self.window = None

    # run loop for demo
    def run(self):

        while True:

            # redraw if needed
            if self.window is None:
                self.redraw_window()
                cv2.imshow(WINDOW_TITLE, self.window)

            # handle keyboard
            k = cv2.waitKey(5)

            if k == 27:
                break
            elif k == ord('l') or k == ord('L'):
                self.show_label = not self.show_label
                self.window = None
            elif k == ord('i') or k == ord('I'):
                self.interactive = not self.interactive
                self.reset_interactive()
            elif k == ord(' '):
                if not self.interactive:
                    self.set_cur_idx(self.cur_idx + 1)
                else:
                    self.reset_interactive()


######################################################################
# main function
        
if __name__ == '__main__':
    demo = NNetDemo()
    demo.run()


    
