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

def load_summit_centered_peaks(
        original_peaks, half_peak_width, max_n_peaks=None):
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

chipseq_peaks_tabix_file_cache = {}
def classify_chipseq_peak(self, chipseq_peaks_fnames, peak):
    pid = os.getppid()
    peak_coords = (peak.contig, 
                   peak.start, 
                   peak.stop)
    status = []
    for motif in self.motifs:
        motif_status = []
        for fname in chipseq_peak_filenames:
            # get the tabix file pointer, opening it if necessary 
            try: 
                fp = tabix_file_cache[(pid, fname)]
            except KeyError:
                fp = TabixFile(fname)
                self.tabix_file_cache[(pid, fname)] = fp

            # if the contig isn't in the contig list, then it
            # can't be a vlaid peak
            if peak[0] not in fp.contigs: 
                motif_status.append(0)
                continue
            overlapping_peaks = list(fp.fetch(peak_coords))
            if len(pc_peaks) > 0:
                motif_status.append(1)
                continue
            else:
                motif_status.append(0)
        status.append(motif_status)
    return status
