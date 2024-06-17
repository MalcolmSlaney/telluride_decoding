import math

from dataclasses import dataclass
from typing import List, Optional, Tuple

from absl import logging
import numpy as np
import pylsl


########################## Data and Time Streams #####################

class DataStream(object):
  """A circular buffer for storing data that is coming in via a stream and for
  which we want to access some amount of data from the past.
  """
  def __init__(self, frame_count:int, dtype: type = float):
    self._data = None  # num_frames x num_dims
    self._frame_rate = None
    self._buffer_count = frame_count     # How big is the buffer?
    self._buffer_time = 0   # How many frames have been stored in the buffer?
    self._buffer_index = 0  # Where are we inserting new frames into the buffer?
    self._dtype = dtype

  def _create_buffer(self, num_dims: int):
    if self._data == None:
      self._data = np.zeros((self._buffer_count, num_dims), dtype=self._dtype)
      self._buffer_index = 0

  def add_data(self, new_data: np.ndarray):
    if self._data is None:
      self._create_buffer(new_data.shape[1])

    assert new_data.shape[0] <= self._buffer_count

    # How much space is left at the end
    frames_to_end = self._buffer_count - self._buffer_index
    # Copy what we can
    end_buffer_count = min(new_data.shape[0], frames_to_end)
    end_frame = self._buffer_index+end_buffer_count
    self._data[self._buffer_index:end_frame, :] = new_data[:end_buffer_count, :]
    self._buffer_index = end_frame

    # How many didn't fit at the end?   Wrap them around to the beginning.
    frames_to_copy = new_data.shape[0] - end_buffer_count
    assert frames_to_copy >= 0
    if frames_to_copy > 0:
      self._data[0:frames_to_copy, :] = new_data[-frames_to_copy:, :]
      self._buffer_index = frames_to_copy
    self._buffer_time += new_data.shape[0]

  def get_data(self, frame_time: int, frame_count: int):
    if not self._buffer_count:
      return None
    if frame_time >= self._buffer_time:
      return None  # Too far in the future
    if frame_time < self._buffer_time - self._buffer_count:
      return None  # Too far in the past

    buffer_start = frame_time % self._buffer_count
    if buffer_start >= self._buffer_index:
      buffer_end = min(self._buffer_count, self._buffer_index + frame_count)
      first_part = self._data[buffer_start:buffer_end, :]
      frame_count -= first_part.shape[0]
      buffer_start = 0
    else:
      first_part = None

    if frame_count > 0:
      frame_count = min(frame_count, self._buffer_index)
      second_part = self._data[buffer_start:frame_count, :]
      if first_part is None:
        return second_part
      return np.concatenate((first_part, second_part), axis=0)
    return first_part
  

class TimeStream(DataStream):
  """A refinement of DataStream, but this one keeps track of the sample rate
  so you request new data by time.  (And when you insert new data, you also 
  provide a time and it checks to make sure there aren't any gaps.)
  """
  def __init__(self, sample_rate: float, buffer_count: Optional[int] = None, 
               name:str = '', dtype=float):
    self._name = name
    if sample_rate <= 0:
      logging.error(f'Sample rate for {self._name} TimeStream can not '
                    f'be {sample_rate}')
    self._sample_rate = float(sample_rate)
    self._start_time = 0
    self._end_time = 0
    buffer_count = buffer_count or int(sample_rate)
    super().__init__(buffer_count, dtype=dtype)

  def get_data_at_time(self, time: float, frame_count: int):
    return super().get_data(math.floor(time*self._sample_rate), frame_count)

  def add_data_at_time(self, data, timestamp):
    if self._start_time == 0:
      self._start_time = timestamp
      self._end_time = timestamp
    delta_samples = (timestamp - self._end_time)/self._sample_rate
    if delta_samples < -0.5:
      logging.warning(f'TimeStream {self._name}: Adding data '
                      f'{delta_samples} samples before the end.')
    if delta_samples > 1.5:
      logging.warning(f'TimeStream {self._name}: Adding data '
                      f'{delta_samples} samples gap.')
    self.add_data(data)
    self._end_time += data.shape[0]/self._sample_rate
    # print(f'{self._name}: {self._start_time}, {self._end_time}, {data.shape}')
  
  @property
  def sample_rate(self):
    return self._sample_rate

  @property
  def start_time(self):
    return self._start_time
  
  @property
  def end_time(self):
    return self._end_time


def end_stream_time(time_streams: List[TimeStream]):
  """Go through all the listed TimeStream objects and retrieve the latest time
  for which all streams have good data."""
  return min([ts.end_time for ts in time_streams])


##############  Python Lab Stream Layer #################################
def read_chunks(inlet):
  chunk_count = 0
  all_timestamps = []
  start_time = 0
  while True:
    # get a new sample (you can also omit the timestamp part if you're not
    # interested in it)
    # sample, timestamp = inlet.pull_sample()  # pull_chunk
    # print(timestamp, sample)
    chunks, timestamps = inlet.pull_chunk()
    for chunk, timestamp in zip(chunks, timestamps):
      if not start_time:
        print(f'Starting chunk list at time {timestamp}')
        start_time = timestamp
      timestamp -= start_time
      all_timestamps.append(timestamp)
      print(timestamp, chunk)
      chunk_count += 1
      if chunk_count > 10:
          break
    if len(all_timestamps) > 10:
      break

  timestamps = np.asarray(timestamps)
  # print(timestamps[1:]-timestamps[:-1])


def read_from_inlet(inlet, timeout:float = 1) -> Tuple[float, np.ndarray]:
  chunks, timestamps = inlet.pull_chunk(timeout=timeout)
  if timestamps:
    return  timestamps[0], np.asarray(chunks)
  return None, None


def open_stream(name: str, debug: bool = False):
  # first resolve an EEG stream on the lab network
  print(f'\nLooking for a {name} stream...')
  # streams = pylsl.resolve_stream("type", "EEG")  # Replace with EEG for a different channel
  streams = pylsl.resolve_stream("name", name)

  # create a new inlet to read from the stream
  inlet = pylsl.StreamInlet(streams[0])

  if debug:
    # get the full stream info (including custom meta-data) and dissect it
    info = inlet.info()
    print("The stream's XML meta-data is: ")
    print(info.as_xml())
    # print("The manufacturer is: %s" % info.desc().child_value("manufacturer"))
    # print("Cap circumference is: %s" % info.desc().child("cap").child_value("size"))
    print("The channel labels are as follows:")
    ch = info.desc().child("channels").child("channel")
    for k in range(info.channel_count()):
      print(ch.child_value("label"), end=' ')
      ch = ch.next_sibling()
    print('n')
  return inlet

@dataclass
class BrainItems:
  name: str
  lsl: pylsl.StreamInlet
  stream: TimeStream


all_stream_names = ['MyAudioStream', 'actiCHamp-18110006', 'NextSense']


def main():
  print("looking for streams")

  streams = pylsl.resolve_streams()
  # iterate over found streams, creating specialized inlet objects that will
  # handle plotting the data
  for info in streams:
    print(f'Type: {info.type()}, name: {info.name()}, sr={info.nominal_srate()}')
      # if info.type() == "Markers":
      #     if (
      #         info.nominal_srate() != pylsl.IRREGULAR_RATE
      #         or info.channel_format() != pylsl.cf_string
      #     ):
      #         print("Invalid marker stream " + info.name())
      #     print("Adding marker inlet: " + info.name())
      # elif (
      #     info.nominal_srate() != pylsl.IRREGULAR_RATE
      #     and info.channel_format() != pylsl.cf_string
      # ):
      #     print("Adding data inlet: " + info.name())
      # else:
      #     print("Don't know what to do with stream " + info.name())
  print('done listing streams.')

  all_streams = {}
  for name in all_stream_names:
    inlet = open_stream(name)
    info = inlet.info()
    print(f'The {name} sample rate is {info.nominal_srate()}Hz')
    ts = TimeStream(sample_rate=info.nominal_srate(), name=name)
    all_streams[name] = BrainItems(name, inlet, ts)

  for _ in range(10):
    for brain_item in all_streams.values():
      inlet = brain_item.lsl
      ts = brain_item.stream
      while True:
        timestamp, data = read_from_inlet(inlet, timeout=0.01)
        if not timestamp:
          break
        print(f'Read from {name} inlet returned', data.shape, 'at', timestamp)
        endtime = ts.add_data_at_time(data, timestamp)
        break
    print('\n')

  for brain_item in all_streams.values():
    ts = brain_item.stream
    print(f'TimeStream {brain_item.name}')
    print(f'  Sample rate: {ts.sample_rate}')
    print(f'  Total seconds recorded: {ts.end_time - ts.start_time}s')

  all_stream_objects = [bi.stream for bi in all_streams.values()]
  print('Latest stream time is', end_stream_time(all_stream_objects))

if __name__ == '__main__':
  main()
