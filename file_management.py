from ast import arg
from cmath import log10
from types import LambdaType
import scipy.io.wavfile as wav
import os
import re
import numpy as np
import utils
import random
from sklearn.metrics import mean_squared_error
from itertools import combinations
import logging
from alive_progress import alive_bar, config_handler
config_handler.set_global(
    title_length = 30
)

MonoAudioBatch = list[tuple[int, np.ndarray]]
UniformAudioBatch = tuple[int, list[np.ndarray]]
AudioMatrix = tuple[int, np.ndarray]

# change this if the random choices display some quirks
_RNG_SEED = 1234

# enable debug printing and assertions
_DEBUG = False
_DEBUG_VERBOSE = False

# setup logger
_logger = logging.getLogger(__name__)
_FORMAT = "[%(filename)s:%(lineno)s - %(funcName)s()]: %(message)s"
logging.basicConfig(format=_FORMAT)
_logger.setLevel(logging.DEBUG)
_logger.disabled = not _DEBUG
# TODO: switch to using logger across entire file


def _read_wav_batch(dir_path: str, channel: str = 'm') -> MonoAudioBatch:
    """get a list of wav files in `dir_path` as mono float buffs
    
    ## Parameters:
    - `dir_path`: path to directory containing batch
    - `channel`: either 'l', 'r', 'm', 's'

    ## Notes: 
    Outputs are not normalized, so if the original file was int,
    the range will be huge.
    """

    # get a list of wav file names in path
    all_files = os.listdir(dir_path)
    wav_files = list(filter(re.compile(r'.*\.wav').match, all_files))
    
    # open files
    with alive_bar(len(wav_files), title='reading files') as bar:
        outs = []
        
        for path in wav_files:
            # stereo data
            fs, data = wav.read(f'{dir_path}/{path}')
            data = data.astype(np.float32)

            if(data.ndim > 1):
                # make into mono
                if channel == 'l':
                    data = data[:,0]
                elif channel == 'r':
                    data = data[:,1]
                elif channel == 'm':
                    data = data[:,0]/2 + data[:,1]/2
                elif channel == 's':
                    data = data[:,0]/2 - data[:,1]/2
                else:
                    _logger.error(utils.dbg_error_formatter(f"unknown channel config: '{channel}'"))
                    exit()
            
            # convert audio data type to float ndarray
            #data = np.array(list(map(float, data)))

            # zero center
            data -= data.mean()

            # normalize
            peak = max(abs(data.max()), abs(data.min()))
            data /= peak
            
            # add to return values
            outs.append((fs, data))

            bar()
    
    return outs


def _batch_conform_sr(batch: MonoAudioBatch, resampler_f: LambdaType) -> UniformAudioBatch:
    """takes a batch and a resampling lambda and produces a new 
    batch where sample rates match
    """
    
    with alive_bar(title='conform sample rates', ) as bar:
    # finds the maximum sample rate
        maxSr = 0
        [
            maxSr := max(maxSr, sr)
            for sr, _ in batch
        ]

        # do upsampling and remove sample rate
        batch = [
            data if sr == maxSr else resampler_f(data, sr, maxSr)
            for sr, data in batch
        ]

    return maxSr, batch


def _batch_trim_silence(batch: UniformAudioBatch, thresh_dbfs: float) -> UniformAudioBatch:
    """trims silence at the beginning of all samples in a batch"""
    
    thresh_amp = utils.dbfs_to_amplitude(thresh_dbfs)
    sr, data = batch
    with alive_bar(len(data), title='trimming silence') as bar:
        for i, clip in enumerate(data):
            trim_idx = 0
            for j, sample in enumerate(clip):
                if abs(sample) < thresh_amp:
                    trim_idx = j
                else:
                    _logger.info(utils.dbg_unimportant_formatter(f'trimming {trim_idx}/{len(clip)} samples')) if _DEBUG_VERBOSE else ()
                    break
            data[i] = clip[trim_idx:]
            bar()
    
    return sr, data


def _batch_conform_duration(batch: UniformAudioBatch, max_duration: float) -> UniformAudioBatch:
    """zero pads the end of samples so they all have the same duration, caps
    at maxDuration"""

    sr, data = batch
    max_len = int(np.ceil(max_duration * sr))

    with alive_bar(title='conform durations') as bar:
        # trim excessively long clips
        data = [
            clip if len(clip) <= max_len else clip[:max_len]
            for clip in data
        ]

        # find longest
        longest = 0
        [
            longest := max(longest, len(clip))
            for clip in data
        ]

        # zero pad shorter ones
        for i, clip in enumerate(data):
            data[i] = np.array(clip)
            data[i].resize(longest)

        if _DEBUG:
            for clip in data:
                assert len(clip) == longest
            longest <= max_len

    return sr, data


def _batch_conform_polarity(batch: UniformAudioBatch) -> UniformAudioBatch:
    """tries to flip polarity to see if it makes samples more similar,
    if not it leaves the polarity unchanged
    
    Notes:
    instead of trying every combination of clips in the batch, it
    selects 5 random clips from the batch, performs a full conform
    on those, then uses them as reference for the rest of the batch
    """
    random.seed(_RNG_SEED)
    sr, data = batch
    references = random.sample(data, 5)

    # fully conform reference clips
    for (i1, clip1), (_, clip2) in combinations(enumerate(references), 2):
        neg_clip1 = -clip1
        rms = mean_squared_error(clip2, clip1, squared=False)
        rms_neg = mean_squared_error(clip2, neg_clip1, squared=False)
        if rms_neg < rms:
            references[i1] = neg_clip1

    # conform rest of clips to reference clips
    with alive_bar(len(data), title='conform polarity') as bar:
        for i, clip in enumerate(data):
            rms_sum = 0
            [
                rms_sum := rms_sum + mean_squared_error(ref_clip, clip, squared=False)
                for ref_clip in references 
            ]
            neg_rms_sum = 0
            [
                rms_sum := rms_sum + mean_squared_error(ref_clip, -clip, squared=False)
                for ref_clip in references 
            ]
            if neg_rms_sum < rms_sum:
                data[i] = -clip
            bar()
    
    return sr, data


def _batch2matrix(batch: UniformAudioBatch) -> AudioMatrix:
    """convert audio batch into matrix form"""

    sr, data = batch
    matrix = np.vstack(data)
    _logger.info(f'audio matrix has dimensions: {matrix.shape}')
    return sr, matrix

def _matrix2batch(a_matrix: AudioMatrix) -> UniformAudioBatch:
    """converts an audio matrix into a uniform batch"""

    sr, matrix = a_matrix
    with alive_bar(title='de-numpyification'):
        data = matrix.tolist()

    with alive_bar(len(data), title='matrix to batch') as bar:
        for i, clip in enumerate(data):
            data[i] = np.array(clip)
            bar()
    return sr, data


def _batch_unpack_sr(batch: UniformAudioBatch) -> MonoAudioBatch:
    """takes a uniform batch and adds explicit sample rate to each clip,
    i.e. it turns it into a non-uniform mono audio batch"""

    sr, data = batch
    return [
        (sr, clip)
        for clip in data
    ]


# FIXME: scipy wav writes a bad header, so it can't be interpreted as stereo
def _write_wav_batch(batch: MonoAudioBatch, dir_path: str, prefix='', to_stereo=False):
    """takes a non-uniform mono batch and writes it to a series of wav files,
    optionally it can copy the mono channel to two stereo channels"""

    max_file_num = len(batch) - 1
    zeros = int(np.ceil(np.log10(max_file_num)))

    with alive_bar(len(batch), title='writing files') as bar:
        for i, (sr, data) in enumerate(batch):
            wav.write(f'{dir_path}/{prefix}{i:0{zeros}}.wav', sr, data if not to_stereo else np.hstack([data, data]))
            bar()

def import_wav_dataset(dir_path: str, to_mono_method: str = 'm', silence_thresh: float = -50, max_len: float = 1) -> AudioMatrix:
    """loads a batch of wav files from a specified directory as a matrix of floats
    to be used in machine learning stuff
    
    ## Parameters:
    - `dir_path`: directory where to look for files
    - `to_mono_method`: either 'l','r','m' or 's', is the method used for turning
    stereo into mono
    - `silence_thresh`: in decibells full-scale is the threshold at which leading silence is trimmed
    - `max_len`: is the maximum length in seconds, after which the remainder is trimmed

    ## Notes:
    May be buggy if input files are not stereo
    """
    _logger.info('importing wav files...')
    data = _read_wav_batch(dir_path, channel=to_mono_method)
    _logger.info('pre-processing audio batch...')
    data = _batch_conform_sr(data, utils.ez_resampler)
    data = _batch_trim_silence(data, silence_thresh)
    data = _batch_conform_duration(data, max_len)
    data = _batch_conform_polarity(data)
    _logger.info('batch to matrix...')
    data = _batch2matrix(data)
    _logger.info('finished importing wav files!')
    return data

# TODO: to_stereo is broken (see the FIXME above _write_wav_batch)
def export_wav_dataset(data: AudioMatrix, dir_path: str, prefix: str = '', to_stereo: bool = False):
    """writes a dataset as a batch of wav files, they are automatically numbered,
    but you can specify a prefix and optionally store as stereo (dual mono)
    
    Parameters:
    - `data`: the data matrix to be stored
    - `dir_path`: destination directory
    - `prefix`: string added to beginning of all file names (before number)
    - `to_stereo`: DOESN'T WORK, supposed to store as stereo file
    """
    _logger.info('matrix to batch...')
    sr, data = _matrix2batch(data)
    with alive_bar(len(data), title='normalization') as bar:
        for clip in data:
            peak = max(abs(clip.max()), abs(clip.min()))
            clip /= peak
            bar()
    data = _batch_unpack_sr((sr, data))
    _logger.info('exporting wav files...')
    _write_wav_batch(data, dir_path=dir_path, prefix=prefix, to_stereo=to_stereo)
    _logger.info('finished exporting wav files!')


# testing
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    matrix = import_wav_dataset('training_data')
    export_wav_dataset(matrix, 'generated_data')

    """
    test_data = _read_wav_batch('training_data')
    pre = test_data
    pre = list(map(lambda pair: pair[1], pre))[0]
    test_data = _batch_conform_sr(test_data, utils.ez_resampler)
    test_data = _batch_trim_silence(test_data, -50)
    test_data = _batch_conform_duration(test_data, 1)
    test_data = _batch_conform_polarity(test_data)
    matrix = _batch2matrix(test_data)
    outs = _matrix2batch(matrix)
    _, post = outs
    post = post[0]
    outs = _batch_unpack_sr(outs)
    _write_wav_batch(outs, 'generated_data', prefix='lel')

    plt.plot(pre)
    #plt.plot(post)
    plt.show()
    """