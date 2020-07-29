"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
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
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

import numpy as np

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

# def preprocessing(input_image, height, width):
#     image = np.copy(input_image)
#     #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     image = cv2.resize(image, (width, height))
#     image = image.transpose((2, 0, 1))
#     image = image.reshape(1, 3, height, width)
#     return image

# def revert_preprocessing(input_image, height, width):
#     image = np.copy(input_image)
#     image = image[0]
#     image = image.transpose((1, 2, 0))
#     image = cv2.resize(image, (width, height))
#     return image

def preprocessing(input_image, height, width):
    image = np.copy(input_image)
    #print('input_image.shape = {}'.format(input_image.shape))
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2, 0, 1))
    image = image.reshape(1, 3, height, width)
    return image

def seconds_to_time(seconds):
    t = seconds
    cents = t - int(t)
    cents = int(cents * 100)
    t = int(t)
    hours = int(t / 60 / 60)
    t -= hours * 60 * 60
    minutes = int(t / 60)
    t -= minutes * 60
    secs = t
    if hours > 0:
        return '{}:{:02d}:{:02d}.{:02d}'.format(hours, minutes, secs, cents)
    return '{:02d}:{:02d}.{:02d}'.format(minutes, secs, cents)

log_file = ''

def log(line, write_on_file = True):
    global log_file
    print(line)
    if not write_on_file: return
    with open(log_file, 'a') as f:
        f.write(line + '\n')
        
def clear_log():
    global log_file
    with open(log_file, 'w') as f:
        f.write('')
        
def get_output_video_file(input_file):
    slash_index = input_file.rfind('/')
    if slash_index == -1: return 'out_{}'.format(args.input)
    return 'out_{}'.format(input_file[slash_index + 1:])

def draw_text(image, text, bottom_left):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (0, 255, 0)
    lineType = 2
    cv2.putText(image, text, bottom_left, font, fontScale, fontColor, lineType)

def process_output(frame, video_size, output, frame_index, time, people_count, prob_threshold):
    #print('output.shape: {}'.format(output.shape))
    n = output.shape[2]
    green_color = (0, 255, 0)
    confidence_threshold = 0.00001 #0.1 #0.2 #0.33 #0.5
    people_detected = False
    for i in range(n):
        box = output[0, 0, i]
        if box[1] == 1 and box[2] > prob_threshold: 
            #log('{}, frame={}, time={}'.format(box, frame_index, time))
            start_point = (int(box[3] * video_size[0]), int(box[4] * video_size[1]))
            end_point = (int(box[5] * video_size[0]), int(box[6] * video_size[1]))
            frame = cv2.rectangle(frame, start_point, end_point, green_color, 2)
            people_detected = True
    if people_count > -1:
        draw_text(frame, 'People count: {}'.format(people_count), (10, 30))
    return frame, people_detected

def get_file_extension(file):
    return file[file.rfind('.') + 1:]

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    
    # Initialise the class
    network = Network(model = args.model, device = args.device, cpu_extension = args.cpu_extension)
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    #print('model={}'.format(args.model))
    network.load_model()

    ### TODO: Handle the input stream ###
    #print('input={}'.format(args.input))
    global log_file
    log_file = '{}.txt'.format(args.input)
    video_file = args.input
    extension = get_file_extension(video_file).lower()
    if video_file.upper() == 'CAM':
        infer_on_video(args, client, 0, network)
    elif extension in ['mp4', 'mov']:
        infer_on_video(args, client, video_file, network)
    elif extension in ['jpeg', 'jpg', 'png', 'bmp']:
        infer_on_image(args, client, video_file, network)
    else:
        print('The extension \"{}\" of your input file \"{}\" is not supported.'.format(extension, video_file))
        exit()


def infer_on_image(args, client, image_file, network):
    input_shape = network.get_input_shape()
    print('input_shape={}'.format(input_shape))
    frame = cv2.imread(image_file, cv2.IMREAD_COLOR)
    input_frame = preprocessing(frame, input_shape[2], input_shape[3])
    network.exec_net(input_frame)
    network.wait()
    output = network.get_output()
    frame_size = frame.shape
    frame_size = (frame_size[1], frame_size[0])
    print('frame_size={}'.format(frame_size))
    output_frame, people_detected = process_output(frame, frame_size, output, 0, 0., -1, args.prob_threshold)
    cv2.imwrite('output_image.png', output_frame)
    
    
def infer_on_video(args, client, video_file, network):
    input_shape = network.get_input_shape()
    #print('input_shape={}'.format(input_shape))
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_period = 1. / frame_rate
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_seconds = n_frames * frame_period
    total_time = seconds_to_time(n_seconds)
    #log('frame_rate={}, frame_period={}, n_frames={}, n_seconds={}, total_time={}'.format(frame_rate, frame_period, n_frames, n_seconds, total_time))
    #log('frame_index, time')
    width = int(cap.get(3))
    height = int(cap.get(4))
    video_size = (width, height)
    output_video = get_output_video_file(args.input)
    out = cv2.VideoWriter(output_video, 0x00000021, frame_rate * 5, (width, height))
    frame_index = 0
    people_timer = 0
    people_timer_count = 20 #10 #5
    presences = []
    person_id = 0
    person_start_frame = -1
    person_end_frame = -1
    client.publish("person", json.dumps({"count": 0}))
    while cap.isOpened():
        seconds = frame_index * frame_period
        flag, frame = cap.read()
        if not flag: break
        key_pressed = cv2.waitKey(60)
        input_frame = preprocessing(frame, input_shape[2], input_shape[3])

        network.exec_net(input_frame)
        network.wait()
        output = network.get_output()
        
        time = seconds_to_time(frame_index * frame_period)
        line = '{}, {}, {}, {}'.format(frame_index, time, person_id, people_timer)
        #print(line)
        output_frame, people_detected = process_output(frame, video_size, output, frame_index, time, person_id, args.prob_threshold)
        
        if people_detected:
            people_timer = int(people_timer_count)
        else:
            people_timer -= 1
        if people_timer < 0: people_timer = 0
        if people_timer > 0 and person_start_frame == -1:
            person_id += 1
            person_start_frame = frame_index
            client.publish("person", json.dumps({"total": person_id}))
            client.publish("person", json.dumps({"count": 1}))            
        if people_timer == 0 and person_start_frame >= 0:
            person_end_frame = frame_index - int(people_timer_count - 1)
            presences.append((person_id, person_start_frame, person_end_frame))
            duration = (person_end_frame - person_start_frame) * frame_period
            client.publish("person/duration", json.dumps({"duration": duration}))
            client.publish("person", json.dumps({"count": 0}))
            person_start_frame = -1        
        
        #people_in_frame = 1 if people_timer > 0 else 0
        #client.publish("person", json.dumps({"count": people_in_frame}))        
        
        #cv2.imwrite('FRAME.PNG', output_frame)
        out.write(output_frame)
        sys.stdout.buffer.write(output_frame)
        sys.stdout.flush()
        frame_index += 1
        if key_pressed == 27: break
        #if frame_index > 10: break
    cap.release()
    cv2.destroyAllWindows()
    out.release()
    #print('presences: {}'.format(presences))

    ### TODO: Loop until stream is over ###

        ### TODO: Read from the video capture ###

        ### TODO: Pre-process the image as needed ###

        ### TODO: Start asynchronous inference for specified request ###

        ### TODO: Wait for the result ###

            ### TODO: Get the results of the inference request ###

            ### TODO: Extract any desired stats from the results ###

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###

        ### TODO: Send the frame to the FFMPEG server ###

        ### TODO: Write an output image if `single_image_mode` ###


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
