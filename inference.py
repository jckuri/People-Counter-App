#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self, model, device = "CPU", cpu_extension = None):
        ### TODO: Initialize any class variables desired ###
        self.model = model
        self.device = device
        self.cpu_extension = cpu_extension

    def load_model(self):
        ### TODO: Load the model ###
        ### TODO: Check for supported layers ###
        ### TODO: Add any necessary extensions ###
        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        self.plugin = IECore()
        if self.cpu_extension and "CPU" in self.device:
            self.plugin.add_extension(self.cpu_extension, self.device)
        #self.plugin.add_extension('/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/cldnn_global_custom_kernels/prior_box_clustered.cl', self.device)
        model_bin = self.model[:-4] + '.bin'
        self.network = IENetwork(model = self.model, weights = model_bin)
        self.exec_network = self.plugin.load_network(self.network, self.device)
        self.input_data = next(iter(self.network.inputs))
        self.output_data = next(iter(self.network.outputs))
        
        supported_layers = self.plugin.query_network(self.network, self.device)
        non_supported_layers = [layer for layer in self.network.layers.keys() if layer not in supported_layers]
        if len(non_supported_layers) > 0:
            print('The following layers are not supported: {}'.format(non_supported_layers))
            exit()

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        return self.network.inputs[self.input_data].shape

    def exec_net(self, image, request_id = 0):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        self.exec_network.start_async(request_id = request_id, inputs = {self.input_data : image})

    def wait(self, request_id = 0):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return self.exec_network.requests[request_id].wait(-1)
    
    def get_output(self, request_id = 0):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return self.exec_network.requests[request_id].outputs[self.output_data]
