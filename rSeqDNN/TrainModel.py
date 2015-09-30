from pyTFbindtools.peaks import load_summit_centered_peaks, load_narrow_peaks
import argparse
import re
import gzip
from pysam import FastaFile

def BedFileArray2fasta(bed_file_array, fasta_file, fa_dir='../EncodeHg19MaleMirror/'):
    '''
    input: bed file
    output: fasta file with sequences for all chromosomes
    '''
    print 'writing fasta file: ', fasta_file
    tmp_bed_file = fasta_file + '.bed.tmp'
    tmp_fasta_file = fasta_file + '.tmp'
    datalines = bed_file_array
    chr_names = []
    for i in range(1, 23):
        chr_names.append('chr' + str(i))
    for chr_name in chr_names:
        datalines_chr = datalines[datalines[:,0]==chr_name]
        np.savetxt(tmp_bed_file, datalines_chr, fmt='%s', delimiter='\t')
        input_fasta = fa_dir + chr_name +'.fa'
        #!bedtools getfasta -tab -fi $input_fasta -bed $tmp_bed_file -fo $tmp_fasta_file  # create fasta file
        #!cat $tmp_fasta_file >> $fasta_file
    #!rm $tmp_bed_file
    #!rm $tmp_fasta_file

def Fasta2RegionsAndOneHotEncoding(fasta_file):
    '''
    input: fasta file
    output: list regions (chr:start-end) and one hot encoding arrays of sequences
    '''
    datalines = np.loadtxt(fasta_file, dtype='str')
    print "num of sequences in fasta file: ", len(datalines)
    filenames = np.copy(datalines[:, 0])
    filenames = filenames.tolist()
    sequences = np.copy(datalines[:, -1])
    print 'processing fasta strings...'
    seq_Strings = []
    for seq in sequences:
        if set(seq) <= set('ACGTacgtN'):
            seq_Strings.append(seq.upper())
    print 'num of sequences: ', len(seq_Strings)
    seq_Lines = []
    for seq in seq_Strings:
        seq_Lines.append(list(seq))
    print 'creating one hot encodings...'
    seq_Array = np.asarray(seq_Lines)
    N, k = np.shape(seq_Array)
    one_hot_encoding = np.zeros((N, 4, k), dtype='bool')
    letters = ['A', 'C', 'G', 'T']
    for i in xrange(len(letters)):
        letter = letters[i]
        one_hot_encoding[:, i, :] = seq_Array == letter
    
    return filenames, one_hot_encoding


def peaks_to_pngs(peaks,fasta):
    '''Covert a list of peaks into pngs encoding their sequence. 
    
    input: peaks list, fasta file
    output: list of png filenames
    note: currently doesn't write reverse complements, that can be easily appended to this function
    '''
    fasta_file = filename + '.fa'
    #!rm $fasta_file  # remove fasta file in case file with the same already exists
    symmetric_bed_array = SymmetricBedArray(bed_array, length)
    BedFileArray2fasta(symmetric_bed_array, fasta_file)
    start_ind = np.copy(np.asarray(symmetric_bed_array[:, 1], dtype=int))
    end_ind = np.copy(np.asarray(symmetric_bed_array[:, 2], dtype=int))
    ps_ind = (end_ind - start_ind) / 2
    print 'getting region names and sequence one hot encodings..'
    filenames, one_hot_encoding = Fasta2RegionsAndOneHotEncoding(fasta_file)
    print 'renaming regions to chr:start-end_ps400.png format...'
    for i in xrange(len(filenames)):
        filenames[i] = filenames[i] + '_ps' + str(ps_ind[i]) + '.png'  
    print 'writing sequences pngs and filenames to files...'
    perBatch = len(filenames) / 10
    file_list = filename + '.filelist'
    with open(file_list, 'w') as wf:
        print 'writing original sequences...'
        count = 0 
        batch = 0
        for i in xrange(len(one_hot_encoding)): ## originals
            filename = filenames[i]
            arr = one_hot_encoding[i]
            misc.imsave(folder_path + filename, arr, format='png')
            wf.write(filename + '\n')
            count += 1
            if count / perBatch > batch:
                batch +=  1
                print 'finished writing ', batch*10,'%' 
                
    return filenames


def getFileHandle(filename,mode="r"):
    if (re.search('.gz$',filename) or re.search('.gzip',filename)):
        if (mode=="r"):
            mode="rb";
        return gzip.open(filename,mode)
    else:
        return open(filename,mode)

def parse_args():
    parser = argparse.ArgumentParser(
        description='main script for training rSeqDNN')
    parser.add_argument('--genome-fasta', type=FastaFile, required=True,
                        help='genome file to get sequences')
    parser.add_argument('--pos-regions', type=getFileHandle, required=True,
                        help='regions with positive label')
    parser.add_argument('--neg-regions', type=getFileHandle, required=True,
                        help='regions with negative labels')
    parser.add_argument('--half-peak-width', type=int, default=400,
                        help='half peak width about summits for training')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    genome_fasta = args.genome_fasta
    pos_peaks = load_summit_centered_peaks(
        load_narrow_peaks(args.pos_regions), args.half_peak_width)
    neg_peaks = load_summit_centered_peaks(
        load_narrow_peaks(args.neg_regions), args.half_peak_width)


if __name__ == '__main__':
    main()
