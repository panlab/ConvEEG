The EEG data were recorded with a 32-channel BrainCap MR using a 32-channel Brain-Amp Amplifier (Brain Products, Munich, Germany, 5 kHz sampling). 
  
Inside a P_(participant id) compressed file:
• (Trial id).txt
• MemoryTest.txt

Format of (Trial id).txt:
  750-by-30 matrix in ASCII-delimited format of MATLAB
  one line for one row
  elements in the same row are separated by commas
  750 rows for 750 time samples
  30 columns for 30 channels
  frequency: 250Hz

Format of MemoryTest.txt:
  N-by-1 matrix in ASCII-delimited format of MATLAB
  one line for one element
  each element indicates the result of memory test for each trial, that is “remembered” or “forgotten”. “1” represents     
  “remembered” and “0” represents “forgotten”.
  N elements for N trials in sequential order
  
Format of Channel.txt:
  C-by-1 matrix in ASCII-delimited format of MATLAB
  one line for one element
  each element indicates the name of channel used for data collection
  C elements for C channels in sequential order, which is consistent with the channel order in (Trial id).txt
