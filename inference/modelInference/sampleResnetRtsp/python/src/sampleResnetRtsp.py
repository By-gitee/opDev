import numpy as np
import videocapture as video

from acllite_resource import AclLiteResource
from acllite_model import AclLiteModel
from acllite_imageproc import AclLiteImageProc
from label import label
from acllite_logger import log_error, log_info


class SampleResnetRtsp(object):
    '''load the model, and do preprocess, infer, postprocess'''
    def __init__(self, model_path, model_width, model_height):
        # initial parameters
        self.model_path = model_path
        self.model_width = model_width
        self.model_height = model_height
        
    def init_resource(self):
        # initial acl resource, create image processor, create model
        self._resource = AclLiteResource()
        self._resource.init()
    
        self._dvpp = AclLiteImageProc(self._resource) 
        self._model = AclLiteModel(self.model_path)

    def preprocess(self, image):
        # process frames by AclLiteImageProc
        self.resize_image = self._dvpp.resize(image, self.model_width, self.model_height)
    
    def infer(self):
        # add input image to list and inference
        self.output = self._model.execute([self.resize_image])
    
    def postprocess(self):
        # decode the result and print top 1 
        infer_out = np.array(self.output, dtype=np.float32).flatten()
        result = np.exp(infer_out) / np.sum(np.exp(infer_out))
        top_index = result.argsort()[-1]
        label_string = label.get(str(top_index))
        label_format = ', '.join(label_string)
        log_info(f"top 1: index[{top_index}] value[{result[top_index]:.6f}] class[{label_format}]")

    def release_resource(self):
        # release resource includes acl resource, data set and unload model
        del self._resource
        del self._dvpp
        del self._model
        del self.resize_image


if __name__ == '__main__':  
    stream_path = "rtsp://192.168.192.98:8554/h264ESVideoTest"
    model_path = "../model/resnet50.om"
    model_width = 224
    model_height = 224

    # create rtsp and init acl resource , load model
    rtsp = SampleResnetRtsp(model_path, model_width, model_height)
    ret = rtsp.init_resource()
    cap = video.VideoCapture(stream_path)

    while True:
        ret, image = cap.read()
        if ret:
            log_error("vdec read from channel failed")
            break
        if image is not None:
            rtsp.preprocess(image)
            rtsp.infer()
            rtsp.postprocess()
        else:
            log_info("read frame finish")
            break
    rtsp.release_resource()