import gzip

from collections import namedtuple

NarrowPeakData = namedtuple(
    'NarrowPeak', ['contig', 'start', 'stop', 'summit', 'score'])

class NarrowPeak(NarrowPeakData):
    @property
    def identifier(self):
        return "{0.contig}:{0.start}-{0.stop}_{0.summit}".format(self)
    @property    
    def pk_width(self):
        return self.stop - self.start

class Peaks(list):
    pass

def getFileHandle(filename, mode="r"):
    if filename.endswith('.gz') or filename.endswith('.gzip'):
        if (mode=="r"):
            mode="rb";
        return gzip.open(filename,mode)
    else:
        return open(filename,mode)


def load_summit_centered_peaks(original_peaks, half_peak_width, max_n_peaks=None):
    peaks = Peaks()
    for peak in original_peaks:
        centered_peak = NarrowPeak(
            peak.contig, 
            peak.start+peak.summit-half_peak_width, 
            peak.start+peak.summit+half_peak_width,
            half_peak_width, 
            -1)
        # skip peaks that are too close to the contig start
        if centered_peak.start <= 0: continue
        peaks.append(centered_peak)
    return peaks

def load_narrow_peaks(fp, max_n_peaks=None):
    if isinstance(fp, str):
        raise ValueError, "Expecting filepointer"
    peaks = Peaks()
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
        except IndexError: summit = (stop-start)/2
        peaks.append(NarrowPeak(chrm, start, stop, summit, score))

    return peaks
