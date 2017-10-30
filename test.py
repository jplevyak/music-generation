import numpy as np
from data_utils.parse_files import *

def test(filename, filename2):
    data, bitrate = read_wav_as_np(filename)
    time_blocks = convert_np_audio_to_sample_blocks(data, 1024)
    print('time_blocks', len(time_blocks), len(time_blocks[0]))
    ft_blocks = time_blocks_to_fft_blocks(time_blocks, count=1)
    print('ft_blocks', len(ft_blocks), len(ft_blocks[0]))
    ft_blocks = transform(ft_blocks)
    time_blocks = fft_blocks_to_time_blocks(ft_blocks)
    print('time_blocks', len(time_blocks), len(time_blocks[0]))
    song = convert_sample_blocks_to_np_audio(time_blocks)
    print(song.shape)
    write_np_as_wav(song, bitrate, filename2)
    return


# Insert your code here to train the model.
# Note: here we are just using the FFT of the original as a standin for the
# 'model'.
def train(a):
    return a


# Insert your code here to generate a sample from the model.
# Note: here we are just using the FFT of the original as a standin for the
# 'model'.
def generate(a):
    return a


def transform(ft_blocks):
    # convert to (X, Y, 3)
    x = len(ft_blocks)
    y = len(ft_blocks[0])
    print('transform', x, y, type(ft_blocks[0][0]))

    # transform to black and white image
    a = np.asarray(ft_blocks)
    a = np.stack((a, a, a), axis=-1)
    print(a.shape)


    model = train(a)

    a = generate(model)

    # transform from black and white image
    b = np.mean(a, axis=2)
    print(b.shape)

    return b

 
test("datasets/YourMusicLibrary/wave/Happy.wav", "datasets/YourMusicLibrary/wave/Happy2.wav")
