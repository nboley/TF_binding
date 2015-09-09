
class Peaks(list):
    pass

def load_narrow_peaks(fname):
    peaks = Peaks()
    with gzip.open(fname) as fp:
        for i, line in enumerate(fp):
            if line.startswith("track"): continue
            if i > MAX_N_PEAKS: break
            data = line.split()
            chrm = data[0]
            start = int(data[1])
            stop = int(data[2])
            summit = int(data[9])
            peaks.append((chrm, start, stop, summit))
            #proc_queue.put((chrm, start+summit-50, start+summit+50, summit))
    return peaks
