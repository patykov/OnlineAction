import caffe2.python.onnx.frontend
import onnx
from caffe2.proto import caffe2_pb2

# We need to provide type and shape of the model inputs,
# see above Note section for explanation
data_type = onnx.TensorProto.FLOAT
data_shape = (32, 3, 224, 224)
value_info = {
    'data': (data_type, data_shape)
}

predict_net = caffe2_pb2.NetDef()
with open('predict_net.pb', 'rb') as f:
    predict_net.ParseFromString(f.read())

init_net = caffe2_pb2.NetDef()
with open('init_net.pb', 'rb') as f:
    init_net.ParseFromString(f.read())

onnx_model = caffe2.python.onnx.frontend.caffe2_net_to_onnx_model(
    predict_net,
    init_net,
    value_info,
)

onnx.checker.check_model(onnx_model)
