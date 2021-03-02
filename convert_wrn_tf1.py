from copy import deepcopy
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torchvision
import numpy as np

import io
import numpy as np

import torch.utils.model_zoo as model_zoo
import torch.onnx

logger = logging.getLogger(__name__)

from torchvision import datasets
from torchvision import transforms
import torchvision

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# from model_ema import ModelEMA
from model_wrn import WideResNet, build_wideresnet

torch_model = build_wideresnet("cifar10", 10, 0, -1)
checkpoint = torch.load('./checkpoint/cifar10@4000.5_best.pth.tar', map_location="cpu")

torch_model.load_state_dict(checkpoint['avg_state_dict'])

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

def get_cifar10(data_path):
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    test_dataset = datasets.CIFAR10(data_path, train=False, transform=transform_val, download=False)
    return test_dataset

batch_size = 10
test_dataset = get_cifar10('./data')

torch_model.eval()

x = []
for i in range(batch_size):
    x.append(test_dataset[i][0].unsqueeze(0))
x = torch.cat(x, dim=0)

torch_out = torch_model(x)
torch.onnx.export(torch_model,               
                  x,                       
                  "torch_mpl.onnx",
                  export_params=True,
                  opset_version=10,
                  do_constant_folding=True,
                  input_names = ['input'],
                  output_names = ['output'],
                  dynamic_axes={'input' : {0 : 'batch_size'},
                                'output' : {0 : 'batch_size'}})

import onnx

onnx_model = onnx.load("torch_mpl.onnx")
onnx.checker.check_model(onnx_model)

import onnxruntime

ort_session = onnxruntime.InferenceSession("torch_mpl.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
print("Exported model has been tested with ONNXRuntime, and the result looks good!")

from onnx_tf.backend import prepare
onnx_model = onnx.load("torch_mpl.onnx") 
tf_rep = prepare(onnx_model) 
tf_rep.export_graph("tf_model.pb")

import tensorflow.compat.v1 as tf
sess = tf.Session()

with tf.gfile.FastGFile("tf_model.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name="")

input_tensor = graph.get_tensor_by_name('input:0')
output_tensor = graph.get_tensor_by_name('output:0')

with tf.Session(graph=graph) as sess:
    output_vals = sess.run(output_tensor, feed_dict={input_tensor: to_numpy(x)})  #

prediction=int(np.argmax(np.array(output_vals).squeeze(), axis=0))
print(prediction)
