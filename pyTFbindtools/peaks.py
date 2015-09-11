import gzip

from collections import namedtuple

NarrowPeak = namedtuple('NarrowPeak', ['contig', 'start', 'stop', 'summit', 'score'])

class Peaks(list):
    pass

def load_narrow_peaks(fname, max_n_peaks=None):
    peaks = Peaks()
    with gzip.open(fname) as fp:
        for i, line in enumerate(fp):
            if line.startswith("track"): continue
            if max_n_peaks != None and i > max_n_peaks: 
                break
            data = line.split()
            chrm = data[0]
            start = int(data[1])
            stop = int(data[2])
            try: score = float(data[6])
            except IndexError: score = -1
            try: summit = int(data[9])
            except IndexError: summit = start + (stop-start)/2
            peaks.append(NarrowPeak(chrm, start, stop, summit, score))
            #proc_queue.put((chrm, start+summit-50, start+summit+50, summit))
    return peaks
