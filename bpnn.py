# Ananta Dwi Prasetya Purna Yuda
# 05111740000029

import math
import json

TYPE_FINAL_NEURON = 4
TYPE_NEURON = 2
TYPE_INPUT = 1

target = 1
learning_rate = 1
error = 999            # initial error
threshold = 0.00001    # error threshold
iteration = 0


N = {
    'name' : 'final_neuron',
    'input' : [
        {
            'name' : 'top_neuron',
            'input' : [
                {'name':'inputA', 'weight': 0.1, 'value': 0.1, 'type': TYPE_INPUT},
                {'name':'inputB', 'weight': 0.5, 'value': 0.7, 'type': TYPE_INPUT}
            ],
            'weight' : 0.2,
            'type' : TYPE_NEURON
        },
        {
            'name' : 'bottom_neuron',
            'input' : [
                {'name':'inputA', 'weight': 0.3, 'value': 0.1, 'type': TYPE_INPUT},
                {'name':'inputB', 'weight': 0.2, 'value': 0.7, 'type': TYPE_INPUT}
            ],
            'weight' : 0.1,
            'type' : TYPE_NEURON
        }
    ],
    'type' : TYPE_NEURON|TYPE_FINAL_NEURON
}


def calculate_output(net):
    if net['type'] & TYPE_NEURON:
        input = 0
        for child in net['input']:
            input += calculate_output(child) * child['weight']
        net['_tmp_output'] = 1 / (1 + math.exp(-input))
        return net['_tmp_output']
    elif net['type'] & TYPE_INPUT:
        return net['value']

    
def calculate_error_and_new_weight(net, parent_err = None):
    if net['type'] & TYPE_INPUT:
        net['weight'] += learning_rate * parent_err * net['value']
        return net['weight']
    
    if net['_tmp_output'] is None:
        return None
    
    if net['type'] & TYPE_NEURON:
        if net['type'] & TYPE_FINAL_NEURON:
            net['error'] = (target - net['_tmp_output']) * (1 - net['_tmp_output']) * net['_tmp_output']
        else:
            net['weight'] += learning_rate * parent_err * net['_tmp_output']
            net['error'] = parent_err * net['weight'] * (1 - net['_tmp_output']) * net['_tmp_output']
        for child in net['input']:
            calculate_error_and_new_weight(child, net['error'])
        return net['error']

    
while error <= 0 - threshold or error >= 0 + threshold:
    iteration += 1
    calculate_output(N)
    error = calculate_error_and_new_weight(N)
    
print("Iterasi", iteration)
print("Error", error)
print("Output", N['_tmp_output'])
print("Data", json.dumps(N, sort_keys=True, indent=4))
