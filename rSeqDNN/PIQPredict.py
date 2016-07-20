import csv
import sys
import subprocess
# sys.path.append("..")
# sys.path.append("../../pyDNABinding/")

# import pyximport
# pyximport.install()

# from pyTFbindtools.peaks import (
#     load_labeled_peaks_from_beds, 
#     getFileHandle, 
#     load_chromatin_accessible_peaks_and_chipseq_labels_from_DB,
#     PeaksAndLabelsThreadSafeIterator
# )

from pyTFbindtools.peaks import {
    iter_narrow_peaks,
    load_labeled_peaks_from_beds,
    label_and_score_peak_with_chipseq_peaks
}

# from pybedtools import Interval, BedTool



piq_factor_dict = {}
piq_factor_dict['MAX'] = 'MA0058.1 MAX'
piq_factor_dict['CTCF'] = 'MA0139.1 CTCF'
piq_factor_dict['ARID3A'] = 'MA0151.1 ARID3A'
piq_factor_dict['ATF3'] = ''
piq_factor_dict['BCL11A'] = ''
piq_factor_dict['ARID3A'] = 'MA0151.1 ARID3A'
piq_factor_dict['BCL3'] = ''
piq_factor_dict['BCLAF1'] = ''
piq_factor_dict['BHLHE40'] = ''
piq_factor_dict['BRCA1'] = 'MA0133.1 BRCA1'
piq_factor_dict['CEBPB'] = ''
piq_factor_dict['E2F6'] = ''
piq_factor_dict['EGR1'] = 'MA0162.1 Egr1'
piq_factor_dict['ELF1'] = ''
piq_factor_dict['ETS1'] = 'MA0098.1 ETS1'
piq_factor_dict['FOXA1'] = 'MA0148.1 FOXA1'
piq_factor_dict['GABPA'] = 'MA0062.1 GABPA'
# piq_factor_dict['GABPA'] = 'MA0062.2 GABPA'
piq_factor_dict['GATA2'] = 'MA0036.1 GATA2'
piq_factor_dict['HNF4A'] = 'MA0114.1 HNF4A'
piq_factor_dict['JUN'] = ''
piq_factor_dict['JUND'] = ''
piq_factor_dict['MAFF'] = ''
piq_factor_dict['MAFK'] = ''
piq_factor_dict['MYC'] = 'MA0147.1 Myc'
piq_factor_dict['MXI1'] = ''
piq_factor_dict['NANOG'] = ''
piq_factor_dict['NFYB'] = ''
piq_factor_dict['NRF1'] = ''
piq_factor_dict['REST'] = 'MA0138.1 REST'
# piq_factor_dict['REST'] = 'MA0138.2 REST'
piq_factor_dict['RFX5'] = ''
piq_factor_dict['SIN3A'] = ''
piq_factor_dict['SIX5'] = ''
piq_factor_dict['SP1'] = 'MA0079.1 SP1'
# piq_factor_dict['SP1'] = 'MA0079.2 SP1'
piq_factor_dict['SP2'] = ''
piq_factor_dict['SPI1'] = 'MA0080.1 SPI1'
# piq_factor_dict['SPI1'] = 'MA0080.2 SPI1'
piq_factor_dict['TBP'] = 'MA0108.1 TBP'
piq_factor_dict['TBP'] = 'MA0108.2 TBP'
piq_factor_dict['TBP'] = 'MA0386.1 TBP'
piq_factor_dict['TCF12'] = ''
piq_factor_dict['TCF7L2'] = ''
piq_factor_dict['TEAD4'] = ''
piq_factor_dict['USF1'] = 'MA0093.1 USF1'
piq_factor_dict['USF2'] = ''
piq_factor_dict['YY1'] = 'MA0095.1 YY1'
piq_factor_dict['ZBTB33'] = ''
piq_factor_dict['ZBTB7A'] = ''
piq_factor_dict['ZNF143'] = 'MA0088.1 znf143'

def create_piq_bed_file(score_csv_file, output_peak_file):
    output_file_matrix = []
    with open(score_csv_file, "r") as f_handle:
        score_csv_reader = csv.reader(f_handle)
        with open(output_peak_file,"w+") as output_file:
            peak_csv_writer = csv.writer(output_file, quoting=csv.QUOTE_ALL, delimiter='    ')

            for row in score_csv_reader:
                # Ignore the first row
                if (row[1] == "chr"):
                    continue

                output_row = []
                # Chromosome name
                output_row.append(row[1])
                # Starting position
                output_row.append(int(row[2]))
                # Ending position
                output_row.append(int(row[2])+1)
                # 3 fillers
                output_row.append("")
                output_row.append("")
                output_row.append("")
                # Score
                output_row.append(float(row[6]))
                # Append row to the rest of the rows
                output_file_matrix.append(output_row)

    # Finally write all rows to the output file
    peak_csv_writer.writerows(output_file_matrix)

def main():
    # CSV file with chromosome numbers, scores, etc.
    input_csv_file = sys.argv[1]
    # Output bed file that we need to create for transformation
    # into peak file.
    output_bed_file = sys.argv[2]

    # Convert our score CSV file into a bed file.
    create_piq_bed_file(input_csv_file, output_bed_file)

    # Create a dummy file for holding the negative
    # cases. 
    with open('dummy_neg_file','r+') as f_handle:

        # Convert our bed file into a peaks and labels file.
        peaks_and_labels = load_labeled_peaks_from_beds(
            output_bed_file, f_handle)

        # Iterate through the peaks and lables file and obtain 
        # scores for each peak.
        for pk in iter_narrow_peaks(peaks_and_labels):
            labels, scores = \
            label_and_score_peak_with_chipseq_peaks("Need TF_ID",pk,)

            # TODO: WHAT TO DO WITH LABELS AND SCORES? 


    #     intervals = get_intervals_from_peaks(peaks_and_labels)
    #     bedtool = BedTool(intervals)
    #     merged_bedtool = bedtool.sort().merge()

if __name__ == '__main__':
    main()
