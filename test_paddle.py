#!/usr/bin/env python3
import os
import sys
import cv2
import glob
import argparse
import re
import math
import numpy as np
from rapidfuzz.distance import Levenshtein
import string
import logging
import onnxruntime

# Initialize logging and SummaryWriter modules
def init_logging(argdict, log_dir, log_name):
    """
    Initialize logging and SummaryWriter modules.
    
    Parameters:
    - argdict (dict): Dictionary containing arguments and their values.
    - log_dir (str): Directory where the log file will be saved.
    - log_name (str): Name of the log file.
    
    Returns:
    - logger (logging.Logger): Configured logger object.
    """
    logger = init_logger(log_dir, argdict, log_name)
    return logger

# Initialize a logging.Logger to save all running parameters to a log file
def init_logger(logger_dir, argdict, log_name):
    """
    Initialize a logging.Logger to save all running parameters to a log file.
    
    Parameters:
    - logger_dir (str): Directory where the log file will be saved.
    - argdict (dict): Dictionary containing arguments and their values.
    - log_name (str): Name of the log file.
    
    Returns:
    - logger (logging.Logger): Configured logger object.
    """
    from os.path import join

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    fh = logging.FileHandler(join(logger_dir, log_name), mode='w+')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info("Arguments: ")
    for k in argdict.keys():
        logger.info("\t{}: {}".format(k, argdict[k]))

    return logger
    
# Calculate the difference between two tensors
def diff_tensor(img1,img2):
    """
    Calculate the difference between two tensors and print the max, min, and mean differences.
    
    Parameters:
    - img1 (np.array): First image tensor.
    - img2 (np.array): Second image tensor.
    """
    diff = img1-img2
    print("max:",np.max(diff))
    print("minx:",np.min(diff))
    print("mean:",np.mean(abs(diff)))
    print((img1==img2).all())

# Check the type and dimensions of a tensor
def check_tensor(tensor):
    """
    Check the type and dimensions of a tensor and print them.
    
    Parameters:
    - tensor (np.array or list): Tensor to be checked.
    """

    if isinstance(tensor,list):
        print("Type:",type(tensor))
        print("len:",len(tensor))
    else:
        print("Type:",type(tensor))
        print("Shape:",tensor.shape)
        print("dtype:",tensor.dtype)
        
# Preprocess the input image for OCR
def preprocess(args, file_path, preprocess_results_path):
    """
    Preprocess the input image for OCR.
    
    Parameters:
    - args (dict): Dictionary containing arguments and their values.
    - file_path (str): Path to the input image file.
    - preprocess_results_path (str): Directory where the preprocessed results will be saved.
    
    Returns:
    - padding_im (np.array): Preprocessed image tensor.
    - file_path (str): Path to the input image file.
    """

    file_name =  os.path.splitext(os.path.basename(file_path))[0] 
    img = cv2.imread(file_path)

    # Calculate the aspect ratio of all text bars
    width_list = []
    width_list.append(img.shape[1] / float(img.shape[0])) 

    imgC, imgH, imgW = args['rec_image_shape'][:3] 
    max_wh_ratio = imgW / imgH 
    h, w = img.shape[0:2] 
    wh_ratio = w * 1.0 / h 
    max_wh_ratio = max(max_wh_ratio, wh_ratio) 

    if args['use_max_wh_ratio']: 
        imgW = int((imgH * max_wh_ratio)) 

    h, w = img.shape[:2]
    ratio = w / float(h) 

    if math.ceil(imgH * ratio) > imgW: 
        resized_w = imgW
        
    else:
        resized_w = int(math.ceil(imgH * ratio))
    
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    resized_image = resized_image.transpose((2, 0, 1))
    resized_image = (resized_image - args['mean']) * args['scale']
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image

    npy_save = os.path.join(preprocess_results_path, file_name + '.npy')
    reverse_img_path = os.path.join(preprocess_results_path, file_name + '_RGB' +'.png')
    tensor_save_path = os.path.join(preprocess_results_path, file_name + '_BGR' +'.tensor')
    
    if args['input_npy_save']: 
        padding_im_save = padding_im.copy()
        np.save(npy_save,padding_im_save)

    padding_im_reverse = padding_im.copy()
    padding_im_reverse = (padding_im_reverse) * (1/args['scale']) + args['mean']
    padding_im_reverse = np.transpose(padding_im_reverse,(1,2,0))
    padding_im_reverse = np.round(padding_im_reverse)
    padding_im_reverse = padding_im_reverse.astype(np.uint8)

    if args['preprocess_img_save']:
        if args['save_as_RGB']: 
            padding_im_reverse_RGB = cv2.cvtColor(padding_im_reverse,cv2.COLOR_BGR2RGB)
            cv2.imwrite(reverse_img_path,padding_im_reverse_RGB)
        else:
            cv2.imwrite(reverse_img_path, padding_im_reverse)
    
    if args['input_tensor_save']: 
        padding_im_reverse = np.transpose(padding_im_reverse,(2,0,1))
        padding_im_bgr_tensor = np.reshape(padding_im_reverse, (-1,1)) 
        np.savetxt(tensor_save_path,padding_im_bgr_tensor) 
    
    return padding_im, file_path

# Perform OCR inference using the ONNX model
def ocr_inference(args, model, padding_im, file_path, inference_input_tensor_path, inference_opt_tensor_path):
    """
    Perform OCR inference using the ONNX model.
    
    Parameters:
    - args (dict): Dictionary containing arguments and their values.
    - model (onnxruntime.InferenceSession): Loaded ONNX model.
    - padding_im (np.array): Preprocessed image tensor.
    - file_path (str): Path to the input image file.
    - inference_input_tensor_path (str): Directory where the input tensors for inference will be saved.
    - inference_opt_tensor_path (str): Directory where the output tensors from inference will be saved.
    
    Returns:
    - output (list): Output from the OCR model inference.
    """
    model.get_modelmeta()
    first_input_name = model.get_inputs()[0].name
    first_output_name = model.get_outputs()[0].name
    meta = model.get_modelmeta().custom_metadata_map  # metadata
    if 'stride' in meta:
        stride, names = int(meta['stride']), eval(meta['names'])

    file_name = os.path.splitext(os.path.basename(file_path))[0]
    if len(padding_im.shape) == 3:
        im = np.expand_dims(padding_im, axis=0)  # expand for batch dim
    if args['opt_tensor_save']:
        np.savetxt('{}/{}_input.tensor'.format(inference_input_tensor_path, file_name,),
            padding_im.reshape(-1), fmt='%.6f')

    output = model.run([first_output_name], {
                        first_input_name: im})

    if args['opt_tensor_save']:
        np.savetxt('{}/{}_output_0.tensor'.format(inference_opt_tensor_path, file_name),
                output[0].reshape(-1), fmt='%.6f')

    return output

# Normalize text by removing non-alphanumeric characters and converting to lowercase
def normalize_text(text):
    """
    Normalize text by removing non-alphanumeric characters and converting to lowercase.
    
    Parameters:
    - text (str): Text to be normalized.
    
    Returns:
    - (str): Normalized text.
    """
    text = ''.join(
        filter(lambda x: x in (string.digits + string.ascii_letters), text))
    return text.lower()

# Build and return an ONNX model session
def build_onnx_model(model_file):
    """
    Build and return an ONNX model session.
    
    Parameters:
    - model_file (str): Path to the ONNX model file.
    
    Returns:
    - session (onnxruntime.InferenceSession): Loaded ONNX model session.
    """
    session = onnxruntime.InferenceSession(model_file)
    session.get_modelmeta()
    first_input_name = session.get_inputs()[0].name
    first_output_name = session.get_outputs()[0].name

    meta = session.get_modelmeta().custom_metadata_map  # metadata
    if 'stride' in meta:
        stride, names = int(meta['stride']), eval(meta['names'])
    return session
    
# Reverse the prediction string
def pred_reverse(pred):
    """
    Reverse the prediction string.
    
    Parameters:
    - pred (str): Prediction string to be reversed.
    
    Returns:
    - (str): Reversed prediction string.
    """
    pred_re = []
    c_current = ''
    for c in pred:
        if not bool(re.search('[a-zA-Z0-9 :*./%+-]', c)):
            if c_current != '':
                pred_re.append(c_current)
            pred_re.append(c)
            c_current = ''
        else:
            c_current += c
    if c_current != '':
        pred_re.append(c_current)

    return ''.join(pred_re[::-1])
    
# Add a special character to the dictionary
def add_special_char(dict_character):
    """
    Add a special character to the dictionary.
    
    Parameters:
    - dict_character (list): List of characters.
    
    Returns:
    - (list): Updated list of characters with a special character added.
    """
    dict_character = ['blank'] + dict_character
    return dict_character

# Get ignored tokens for CTC decoding
def get_ignored_tokens():
    """
    Get ignored tokens for CTC decoding.
    
    Returns:
    - (list): List of ignored tokens.
    """
    return [0]  # for ctc blank

# Convert text-index into text-label
def decode(character, text_index, text_prob=None, is_remove_duplicate=False, reverse=False):
    """
    Convert text-index into text-label.
    
    Parameters:
    - character (list): List of characters.
    - text_index (list): List of text indices.
    - text_prob (list, optional): List of text probabilities.
    - is_remove_duplicate (bool, optional): Flag to remove duplicate characters.
    - reverse (bool, optional): Flag to reverse the text.
    
    Returns:
    - result_list (list): List of decoded texts and their average probabilities.
    """
    result_list = []
    ignored_tokens = get_ignored_tokens()
    batch_size = len(text_index)
    for batch_idx in range(batch_size):
        selection = np.ones(len(text_index[batch_idx]), dtype=bool)
        if is_remove_duplicate:
            selection[1:] = text_index[batch_idx][1:] != text_index[
                batch_idx][:-1]
        for ignored_token in ignored_tokens:
            selection &= text_index[batch_idx] != ignored_token

        char_list = [
            character[text_id]
            for text_id in text_index[batch_idx][selection]
        ]
        if text_prob is not None:
            conf_list = text_prob[batch_idx][selection]
        else:
            conf_list = [1] * len(selection)
        if len(conf_list) == 0:
            conf_list = [0]

        text = ''.join(char_list)

        if reverse:  # for arabic rec
            text = pred_reverse(text)

        result_list.append((text, np.mean(conf_list).tolist()))
    return result_list

# Postprocess the OCR predictions
def postprocess(preds, character_dict_path, use_space_char, label=None):
    """
    Postprocess the OCR predictions.
    
    Parameters:
    - preds (np.array): Predictions from the OCR model.
    - character_dict_path (str): Path to the character dictionary file.
    - use_space_char (bool): Flag to use space character.
    - label (np.array, optional): Ground truth labels.
    
    Returns:
    - text (list): List of postprocessed text predictions.
    - label (list, optional): List of decoded labels.
    """
    beg_str = "sos"
    end_str = "eos"
    reverse = False
    character_str = []
    if character_dict_path is None:
        character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
        dict_character = list(character_str)
    else:
        with open(character_dict_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                character_str.append(line)
        if use_space_char:
            character_str.append(" ")
        dict_character = list(character_str)
        if 'arabic' in character_dict_path:
            reverse = True

    dict_character = add_special_char(dict_character)
    dict = {}
    for i, char in enumerate(dict_character):
        dict[char] = i
    character = dict_character

    preds_idx = preds.argmax(axis=2)
    preds_prob = preds.max(axis=2)
    text = decode(character, preds_idx, preds_prob, is_remove_duplicate=True, reverse=False)
    if label is None:
        return text, preds_idx, preds_prob, character
    label = decode(label)
    return text, label

# Postprocess OCR results and save to a file
def ocr_postprocess(args, index, output, file_path, txt_results_path, logger):
    """
    Postprocess OCR results and save to a file.
    
    Parameters:
    - args (dict): Dictionary containing arguments and their values.
    - index (int): Index of the current file being processed.
    - output (list): Output from the OCR model inference.
    - file_path (str): Path to the input image file.
    - txt_results_path (str): Directory where the text results will be saved.
    - logger (logging.Logger): Logger object.
    
    Returns:
    - result (list): List of postprocessed text predictions.
    - preds_idx (np.array): Predicted indices.
    - preds_prob (np.array): Predicted probabilities.
    """
    result, preds_idx, preds_prob, character= postprocess(output[0], args['character_dict_path'], args['use_space_char'], label=None)
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    outputtxtname = os.path.join(txt_results_path, file_name + '.txt')
    with open(outputtxtname,'w') as f:
        f.write("No. {} Predicts of {}\npred txt: {}\npred prob: {}\n".format(index+1, file_path, result[0][0], result[0][1]))
        print("No. {} Predicts of {} pred txt: {} pred prob: {}\n".format(index+1, file_path, result[0][0], result[0][1]))
        logger.info("No. {} Predicts of {}\npred txt: {}\npred prob: {}\n".format(index+1, file_path, result[0][0], result[0][1]))

    return result, preds_idx, preds_prob

# Calculate the recognition metrics
def rec_metric(args, preds, labels, logger):
    """
    Calculate the recognition metrics.
    
    Parameters:
    - args (dict): Dictionary containing arguments and their values.
    - preds (list): List of predicted texts.
    - labels (list): List of ground truth labels.
    - logger (logging.Logger): Logger object.
    """
    correct_num = 0
    all_num = 0
    norm_edit_dis = 0.0
    for i in range(len(preds)):
        pred, pred_conf= preds[i][0]
        target = labels[i]
        
        if args['ignore_space']:
            pred = pred.replace(" ", "")
            target = target.replace(" ", "")
        if args['is_filter']:
            pred = normalize_text(pred)
            target = normalize_text(target)
        norm_edit_dis += Levenshtein.normalized_distance(pred, target)
        if pred == target:
            correct_num += 1
        all_num += 1

    correct_num += correct_num
    all_num += all_num
    norm_edit_dis += norm_edit_dis
    eps=1e-5
    temp_dict = {
        'acc': 1.0 *  correct_num / (all_num + eps),
        'norm_edit_dis': 1 - norm_edit_dis / (all_num + eps)
    }
    
    print("acc: {}\nnorm_edit_dis: {}\n".format(temp_dict['acc'], temp_dict['norm_edit_dis']))
    logger.info("acc: {}\n".format(temp_dict['acc']))
    logger.info("norm_edit_dis: {}\n".format(temp_dict['norm_edit_dis']))

# Main function to run the OCR pipeline
def main(**args):
    """
    Main function to run the OCR pipeline.
    
    Parameters:
    - args (dict): Dictionary containing arguments and their values.
    """
    if not os.path.isdir(args['test_path']):
        raise ValueError("Invalid test intput dir `"+os.path.abspath(args['test_path'])+"`")

    if os.access(args['model_file'], os.R_OK) == 0:
        print('cannot access network binary {}'.format(args['model_file']))
        sys.exit(1)

    if not os.path.exists(args['opts_dir']):
        os.makedirs(args['opts_dir'])

    inference_opt_tensor_path = os.path.join(args['opts_dir'], 'inference_opt_tensor')
    preprocess_results_path = os.path.join(args['opts_dir'], 'pre_res') 
    txt_results_path = os.path.join(args['opts_dir'], 'txt_res') 
    inference_input_tensor_path = os.path.join(args['opts_dir'], 'inference_input_tensor')
    log_dir = os.path.join(args['opts_dir'], 'log') 

    if not os.path.exists(inference_opt_tensor_path):
        os.makedirs(inference_opt_tensor_path)
    if not os.path.exists(preprocess_results_path):
        os.makedirs(preprocess_results_path)
    if not os.path.exists(txt_results_path):
        os.makedirs(txt_results_path)
    if not os.path.exists(inference_input_tensor_path):
        os.makedirs(inference_input_tensor_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    filelist = sorted(glob.glob(args['test_path'] + '/*[.png, .jpg, .PNG]'))

    model = build_onnx_model(args['model_file'])
    logger = init_logging(args, log_dir, log_name='log.txt')
        
    res_list = []
        
    for index, file in enumerate(filelist):
        padding_im, file_path = preprocess(args, file, preprocess_results_path)

        output = ocr_inference(args, model, padding_im, file_path, inference_input_tensor_path, inference_opt_tensor_path)
        result, preds_idx, preds_prob = ocr_postprocess(args, index, output, file_path, txt_results_path, logger)
        res_list.append(result)
    
    print(("Total {} are inferenced\nThe raw output tensor are saved at {}\nThe txt results are saved at{}\n").format(len(res_list), inference_opt_tensor_path, txt_results_path))
    logger.info(("Total {} are inferenced\nThe raw output tensor are saved at {}\nThe txt results are saved at{}\n").format(len(res_list), inference_opt_tensor_path, txt_results_path))

if  __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quadra AI Python Inference API for PaddleOCR demo')
    # Load file
    parser.add_argument("--model_file", type=str,default="./PaddleOCR-512_onnx.onnx", \
                        help='path to NBG model')
    parser.add_argument("--test_path", type=str, default="./icdar_data", \
                        help='path to image folder')
    parser.add_argument('--input_npy_save', dest='input_npy_save', action='store_true',
                        help='Save pre-processed input(float32) as npy (default is False)')
    parser.add_argument('--preprocess_img_save', action='store_true',
                        help='Save pre-processed input')
    parser.add_argument('--input_tensor_save', dest='input_tensor_save', action='store_true',
                        help='Save pre-processed input(Uint8) as bgr+nchw tensor')
    parser.add_argument('--opt_tensor_save', action='store_true',
                        help='Save opt tensor')
    parser.add_argument('--save_as_RGB', dest='save_as_RGB', action='store_true',
                        help='ave preprocessed image as RGB')    # Dirs
    parser.add_argument("--mean", nargs='+', type=int,default=[127.5], 
                        help='value of mean for model')
    parser.add_argument("--scale", type=int, default=0.00784313725490196, 
                        help='value of scale for model')
    parser.add_argument("--rec_batch_num", type=int, default=6, 
                        help='rec_batch_num')
    parser.add_argument("--rec_image_shape", nargs='+', type=int, default=[3,48,512], 
                        help='size of target image')
    parser.add_argument('--use_max_wh_ratio', dest='use_max_wh_ratio', action='store_true',
                        help='use max width and height ratio (default is False)')
    parser.add_argument('--use_space_char', dest='use_space_char', action='store_false',
                        help='use space char (default is true)')
    parser.add_argument("--character_dict_path", type=str, default="./characters.txt", \
                        help='path to character list')
    parser.add_argument('--file_format', '-f', type=str, default='nchw',
                        help='specify the model input format')
    parser.add_argument('--channel_order', '-c', type=str, default='bgr',
                        help='specify the order of channels')
    parser.add_argument('--ignore_space',  action='store_true', default=False,\
                        help='ignore the space during')
    parser.add_argument('--is_filter',  action='store_true', default=False,\
                        help='normalize_text')
    parser.add_argument("--opts_dir", type=str, default="./res", \
                        help='path of outputs files ')
    argspar = parser.parse_args()    

    print("\n### Test PaddleOCR NBG model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    main(**vars(argspar))
