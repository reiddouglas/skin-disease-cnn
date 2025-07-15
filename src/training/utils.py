import math

def conv_output_size(input_size, padding, kernel_size, stride):
    return math.floor((input_size + 2 * padding - kernel_size)/2 + 1)