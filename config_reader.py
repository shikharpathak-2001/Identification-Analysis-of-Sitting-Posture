from configobj import ConfigObj
import numpy as np


def config_reader():
    config = ConfigObj('config')

    munish = config['munish']
    model_id = munish['modelID']
    model = config['models'][model_id]
    model['boxsize'] = int(model['boxsize'])
    model['stride'] = int(model['stride'])
    model['padValue'] = int(model['padValue'])
    # munish['starting_range'] = float(munish['starting_range'])
    # munish['ending_range'] = float(munish['ending_range'])
    # munish['octave'] = int(munish['octave'])
    # munish['use_gpu'] = int(munish['use_gpu'])
    munish['starting_range'] = float(munish['starting_range'])
    munish['ending_range'] = float(munish['ending_range'])
    munish['scale_search'] = map(float, munish['scale_search'])
    munish['thre1'] = float(munish['thre1'])
    munish['thre2'] = float(munish['thre2'])
    munish['thre3'] = float(munish['thre3'])
    munish['mid_num'] = int(munish['mid_num'])
    munish['min_num'] = int(munish['min_num'])
    munish['crop_ratio'] = float(munish['crop_ratio'])
    munish['bbox_ratio'] = float(munish['bbox_ratio'])
    # munish['GPUdeviceNumber'] = int(munish['GPUdeviceNumber'])

    return munish, model


if __name__ == "__main__":
    config_reader()
