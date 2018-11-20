"""Human detector powered by YOLOv2.

Download the model '000300.weights' from OneDrive/Temporary.

To train the model, download: http://pjreddie.com/media/files/darknet19_448.conv.23
"""
import os.path as osp

from grab_screen import grab_screen
from module.yolov2.utils import *
from PIL import Image
import cv2
from module.yolov2.darknet import Darknet
# from save_feature import SaveRandomFeature

from parameters import STACK_NUM
from score_detector import ScoreDetector

yolo_root = osp.join('module', 'yolov2')
namesfile = osp.join(yolo_root, 'data', 'fifa.names')
full_game_weight = osp.join('models', 'human_detector', '000300.weights')
shooting_bronze_weight = osp.join('models', 'human_detector', '000440.weights')
goalkeeper_weight = osp.join('models', 'human_detector', '000290.weights')

def visualize_detection(feature, image, fname=None):
    class_names = load_class_names(namesfile)
    result = plot_boxes_cv2(image, feature, fname, class_names)
    return result

class HumanDetector():
    def __init__(self, mode='full', train_mode='shooting', force_no_head=False, verbose=0):
        self.mode = mode
        self.train_mode = train_mode
        self.force_no_head = force_no_head
        self.verbose = verbose
        self.ymin, self.ymax, self.xmin, self.xmax = 33, 753, 8, 1288

        # Load model from file
        self.cfgfile = osp.join(yolo_root, 'cfg', 'yolo-voc.cfg')
        self.model = Darknet(self.cfgfile)
        if verbose:
            print(str(self.model))
            self.model.print_network()

        # Load pre-trained weight file
        if self.train_mode == 'shooting':
            self.weightfile = shooting_bronze_weight
        elif self.train_mode in ['goalkeeper', 'arena']:
            self.weightfile = goalkeeper_weight
        else:
            self.weightfile = full_game_weight
        self.use_cuda = 1
        
        # Register hook for intermediate results
        if mode == 'shallow':
            self.model.models[7].register_forward_hook(self.hook)
        elif mode == 'deep':
            self.model.models[22].leaky18.register_forward_hook(self.hook)

        self.model.load_weights(self.weightfile)
        if self.use_cuda:
            self.model.cuda()
        print('[Human Detector] Model: %s, weights: %s' % (self.cfgfile, self.weightfile))


    def hook(self, module, input, output):
        self.intermediate_outputs = output


    def detect_raw(self, image):
        """Invoke object detection network on current frame and return the raw result."""
        # Prepare the input image
        resized = cv2.resize(image, (self.model.width, self.model.height))
        # Forward the entire network for detection result
        features = None
        full_result = do_detect(self.model, resized, 0.5, 0.4, use_cuda=self.use_cuda, is_predict_only=True)
        return full_result, features


    def detect(self, image):
        """Invoke object detection network on current frame."""
        # Prepare the input image
        resized = cv2.resize(image, (self.model.width, self.model.height))
        # Forward the entire network for detection result
        full_result = do_detect(self.model, resized, 0.5, 0.4, use_cuda=self.use_cuda, is_predict_only=True, force_no_head=self.force_no_head)

        # print(full_result)
        # print(full_result.shape)

        # Hard code the output shape in training mode
        if (not self.force_no_head) and (self.train_mode in ['shooting', 'goalkeeper', 'arena']):
            full_result_append = np.zeros((3, 7))
            if full_result.shape[0] == 0:
                print('[Human Detector] Warning! No object detected.')
            elif full_result.shape[0] < 3:
                full_result_append[:full_result.shape[0], :] = full_result
            else:
                full_result_append[:, :] = full_result[:3, :]
            full_result = full_result_append
        
        # Determine the desired output
        if self.mode in ['shallow', 'deep']:
            result = np.asarray(self.intermediate_outputs[0].data)
        else:
            result = full_result
        
        # print(full_result.shape, result.shape)

        return full_result.astype(np.float), result.astype(np.float)

        # detector.visualize(full_res, mode='full', image=im, fname=result_path+image)
    def visualize(self, feature, mode=None, image=None, fname=None):
        if not mode:
            mode = self.mode
        if mode == 'full':
            boxes = feature
            class_names = load_class_names(namesfile)
            result = plot_boxes_cv2(image, boxes, fname, class_names)
        elif mode == 'shallow':
            result = self._visualize_feature(feature, 8, 16, fname=fname)
        elif mode == 'deep':
            result = self._visualize_feature(feature, 32, 32, fname=fname)
        else:
            print('Invalid mode:', mode)
            result = None
        
        return result

    def _visualize_feature(self, output, row, col, fname=None):
        h, w = output.shape[1:]
        flatten_feature = np.zeros((row * h, col * w))
        for y in range(row):
            for x in range(col):
                flatten_feature[y*h:(y+1)*h, x*w:(x+1)*w] = output[y*row+x, :, :]
        min_feature = np.min(flatten_feature)
        max_feature = np.max(flatten_feature)
        flatten_feature = (flatten_feature - min_feature) / (max_feature - min_feature)
        # flatten_feature *= 255.
        if fname:
            cv2.imwrite(fname, flatten_feature.astype(np.uint8))
        return flatten_feature

    def _get_screen(self):
        # Get screen-shot and pre-process
        screen = grab_screen(region=None)
        screen = screen[self.ymin:self.ymax, self.xmin:self.xmax]
        return screen

    def forward(self):
        return self.detect(self._get_screen())

    def init_input_window(self):
        if self.mode == 'deep':
            shape = (STACK_NUM * 1024, 13, 13)
        elif self.mode == 'shallow':
            shape = (STACK_NUM * 128, 52, 52)
        else:
            shape = (STACK_NUM * 2)
        input_window = np.zeros(shape=shape)
        # Generate and initialize input window with 10 feature maps
        for i in range(0, STACK_NUM):
            input_window[i, :] = self.forward()
        return input_window

#####
## Process only for test usage

def proc_human_detector(q):
    """Do human detection and put the result in the synchronized queue."""
    try:
        test_vid = 'data\\VID01.mp4'
        detector = HumanDetector(mode='full', train_mode='shooting', verbose=0)
        cap = cv2.VideoCapture(test_vid)
        screen = 1
        while cap.isOpened() and (not screen is None):
            crt = time.time()
            ret, screen = cap.read()
            res = detector.detect(screen)
            vis = detector.visualize(res, screen)
            cv2.imwrite('tmp/%05d.jpg' % int(time.time() * 1e3), vis)
            q.put(res)
            print('Queue length:', q.qsize(), 'Time elapsed: %d ms' % ((time.time() - crt) * 1e3))
        cap.release()
    except:
        print('Error occurred in humen detector process!')

def proc_env(q):
    """The env process."""
    try:
        while True:
            if not q.empty():
                q.get()
                # print('Get value from main process')
    except:
        print('Error occurred in env process!')



if __name__ == '__main__':
    mode = "shallow"

    # example = 'shooting_example.jpg'
    # im_example = cv2.imread(example)
    # print(im_example.shape)
    # print(im_example.dtype)

    detector = HumanDetector(mode=mode, verbose=1)

    save = SaveRandomFeature(mode)
    save.save_feature(detector)
    # save_feature(mode)
