#!/usr/bin/env python3
# Copyright 2021 ETH Zurich and University of Bologna.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# arg1: image size, arg2: filter size

import numpy as np
import sys

def convolve2D(kernel, image, padding):
    # Default stride
    strides = 1

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape  = image.shape[0]
    yImgShape  = image.shape[1]

    # Shape of Output Convolution
    xOutput = xImgShape - xKernShape + 1
    yOutput = yImgShape - yKernShape + 1
    output = np.zeros((xOutput, yOutput))

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * image[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output

def emit(name, array, alignment='8'):
	print(".global %s" % name)
	print(".balign " + alignment)
	print("%s:" % name)
	bs = array.tobytes()
	for i in range(0, len(bs), 4):
		s = ""
		for n in range(4):
			s += "%02x" % bs[i+3-n]
		print("    .word 0x%s" % s)

# Define the filter size and the matrix dimension (max, for now, is 128 64-bit elements)
if len(sys.argv) > 1:
	matrix_width = int(sys.argv[1])
#	assert(matrix_width <= 128), "The width of the image cannot be greater than 128 64-bit \
#	                                  elements. If this is not enough, modify the algorithm."
	filter_size = int(sys.argv[2])
	# Filter size must be odd
	assert(filter_size % 2 == 1), "The filter size must be an odd integer number"
else:
	matrix_width = 64
	filter_size = 3

# 64-bit data
dtype = np.float64

# Input image. Take a square image
M = matrix_width
N = matrix_width
# 3 Channels
CH = 3
padding = int(filter_size/2)
M_pad = M + 2*padding
N_pad = N + 2*padding
assert(M % 4 == 0), "Output image dimension must be divisible by 4, pad the input image accordingly"
assert(N % 4 == 0), "Output image dimension must be divisible by 4, pad the input image accordingly"

image = list()
# Generate a random float64 input padded image
for ch in range(CH):
        image += [np.random.rand(M_pad, N_pad).astype(dtype)]

gen_filter = list()
# Generate a random float64 filter
for ch in range(CH):
        gen_filter += [np.random.rand(filter_size, filter_size).astype(dtype)]

# Create the empty o matrix
empty_o = np.zeros((M, N)).astype(dtype)

# Calculate the output matrix
result = np.zeros((M, N)).astype(dtype)
for ch in range(CH):
        result += convolve2D(gen_filter[ch], image[ch], padding).astype(dtype)

# Calculate a checksum
checksum = np.sum(result, dtype=dtype)

# Print information on display
#print("Image:\n")
#print(image)
#print("Filter:\n")
#print(gen_filter)
#print("Results:\n")
#print(result)
#print("\n")
#print(checksum)

# Print information on file
print(".section .data,\"aw\",@progbits")
emit("M", np.array(M, dtype=np.uint64))
emit("N", np.array(N, dtype=np.uint64))
emit("F", np.array(filter_size, dtype=np.uint64))
emit("CH", np.array(CH, dtype=np.uint64))
emit("i", np.concatenate(image), 'NR_LANES*4')
emit("f", np.concatenate(gen_filter), 'NR_LANES*4')
emit("o", empty_o, 'NR_LANES*4')
emit("golden_o", result, 'NR_LANES*4')
emit("o_checksum", checksum)
