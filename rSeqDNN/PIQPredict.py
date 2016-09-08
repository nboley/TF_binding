import csv
import sys
import subprocess
sys.path.append("..")
# sys.path.append("../../pyDNABinding/")

# import pyximport
# pyximport.install()

# from pyTFbindtools.peaks import (
#     load_labeled_peaks_from_beds, 
#     getFileHandle, 
#     load_chromatin_accessible_peaks_and_chipseq_labels_from_DB,
#     PeaksAndLabelsThreadSafeIterator
# )

from pyTFbindtools.peaks import iter_narrow_peaks
from pyTFbindtools.peaks import load_labeled_peaks_from_beds
from pyTFbindtools.peaks import label_and_score_peak_with_chipseq_peaks
from pyTFbindtools.peaks import load_chromatin_accessible_peaks_and_chipseq_labels_from_DB

from pyTFbindtools.cross_validation import ClassificationResult
import numpy as np

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

model_id_dict = {}
model_id_dict['MAX'] = 'T011266_1.02' # MAX
model_id_dict['CTCF'] = 'T044268_1.02' # CTCF
model_id_dict['ARID3A'] = 'T007405_1.02' # ARID3A 
model_id_dict['ATF3'] = 'T025301_1.02' # ATF3 
model_id_dict['BCL11A'] = 'T044305_1.02' # BCL11A
model_id_dict['BCL3'] = 'T153674_1.02' # BCL3
model_id_dict['BCLAF1'] = 'T153671_1.02' # BCLAF1
model_id_dict['BHLHE40'] = 'T014206_1.02' # BHLHE40
model_id_dict['BRCA1'] = 'T077335_1.02' # BRCA1
model_id_dict['CEBPB'] = 'T025313_1.02' # CEBPB
model_id_dict['E2F6'] = 'T076679_1.02' # E2F6
model_id_dict['EGR1'] = 'T044306_1.02' # EGR1
model_id_dict['ELF1'] = 'T077988_1.02' # ELF1
model_id_dict['ETS1'] = 'T077991_1.02' # ETS1
model_id_dict['FOXA1'] = 'T081595_1.02' # FOXA1
model_id_dict['GABPA'] = 'T077997_1.02' # GABPA 
model_id_dict['GATA2'] = 'T084684_1.02' # GATA2
model_id_dict['HNF4A'] = 'T132746_1.02' # HNF4A
model_id_dict['JUN'] = 'T025316_1.02' # JUN
model_id_dict['JUND'] = 'T025286_1.02' # JUND
model_id_dict['MAFF'] = 'T025320_1.02' # MAFF
model_id_dict['MAFK'] = 'T025323_1.02' # MAFK
model_id_dict['MYC'] = 'T014210_1.02' # MYC
model_id_dict['MXI1'] = 'T014191_1.02' # MXI1
model_id_dict['NANOG'] = 'T093268_1.02' # NANOG
model_id_dict['NFYB'] = 'T153691_1.02' # NFYB
model_id_dict['NRF1'] = 'T153683_1.02' # NRF1
model_id_dict['REST'] = 'T044249_1.02' # REST
model_id_dict['RFX5'] = 'T138768_1.02' # RFX5
model_id_dict['SIN3A'] = 'T153703_1.02' # SIN3A
model_id_dict['SIX5'] = 'T093385_1.02' # SIX5
model_id_dict['SP1'] = 'T044652_1.02' # SP1
model_id_dict['SP2'] = 'T044474_1.02' # SP2
model_id_dict['SPI1'] = 'T077981_1.02' # SPI1
model_id_dict['TBP'] = 'T150542_1.02' # TBP
model_id_dict['TCF12'] = 'T014212_1.02' # TCF12
model_id_dict['TCF7L2'] = 'T144100_1.02' # TCF7L2
model_id_dict['TEAD4'] = 'T151689_1.02' # TEAD4
model_id_dict['USF1'] = 'T014218_1.02' # USF1
model_id_dict['USF2'] = 'T014176_1.02' # USF2
model_id_dict['YY1'] = 'T044261_1.02' # YY1 
model_id_dict['ZBTB33'] = 'T044578_1.02' # ZBTB33
model_id_dict['ZBTB7A'] = 'T044594_1.02' # ZBTB7A
model_id_dict['ZNF143'] = 'T044468_1.02' # ZNF143

def create_piq_bed_file(score_csv_file, original_bed_file, output_peak_file):
    output_file_matrix = []
    with open(score_csv_file, "r") as f_handle:
        score_csv_reader = csv.reader(f_handle)
        # Ignore the first row
        next(score_csv_reader)
        with open(output_peak_file,"w+") as output_file:
            peak_csv_writer = csv.writer(output_file, quoting=csv.QUOTE_NONE, delimiter='\t')
            original_bed_reader = csv.reader(original_bed_file, quoting=csv.QUOTE_NONE, delimiter='\t')
            # Ignore the first row
            next(original_bed_reader)

            for row in original_bed_reader:
                # # Ignore the first row
                # if (row[1] == "chr"):
                #     continue

                print(row)
                score_csv_row = next(score_csv_reader)

                output_row = []
                # Chromosome name
                output_row.append(row[0])
                # Starting position
                output_row.append(int(row[1]))
                # Ending position
                output_row.append(int(row[2]))
                # 3 fillers, dummy values
                output_row.append(row[3])
                # Replace the 5th column with purity
                output_row.append(float(score_csv_row[6]))
                output_row.append(row[5])
                # Append row to the rest of the rows
                output_file_matrix.append(output_row)
                print(output_row)

            # Finally write all rows to the output file
            peak_csv_writer.writerows(output_file_matrix)

def main():
    # CSV file with chromosome numbers, scores, etc.
    input_csv_file = sys.argv[1]
    # Output bed file that we need to create for transformation
    # into peak file.
    output_bed_file = sys.argv[2]
    # Original bed file given by PIQ
    original_bed_file = sys.argv[3]
    # TF name
    TF_name = sys.argv[4]
    TF_id = model_id_dict[TF_name]

    # Convert our score CSV file into a bed file.
    create_piq_bed_file(input_csv_file, original_bed_file, output_bed_file)

    # Create a dummy file for holding the negative
    # cases. 
    with open('dummy_neg_file','w+') as f_handle:
        with open(output_bed_file,'r') as output_handle:
            # Convert our bed file into a peaks and labels file.
            piq_peaks_and_labels = load_labeled_peaks_from_beds(
                output_handle, f_handle,1)

            # FIX R
            # Cell type selection -> parametrizable (need to get right bam files), TF_name parametrizable, + right csv file
            # Method to get BAM files
            # Loop it over cell types, and TFs.
            # TRY MITRA
            
            our_peaks_and_labels = \
            load_chromatin_accessible_peaks_and_chipseq_labels_from_DB( \
                TF_id, \
                1, \
                half_peak_width=500, \
                include_ambiguous_peaks=True)

            # General preprocessing
            our_peaks_and_labels = our_peaks_and_labels.remove_ambiguous_labeled_entries()
            # DO WE NEED THIS?
            # our_peaks_and_labels = our_peaks_and_labels.filter_by_contig_edge(9000, genome_fasta) # removes peaks in contigs edges
            our_peaks_and_labels = our_peaks_and_labels.subset_data(["E116"],our_peaks_and_labels.contigs)

            # NEED TO CONVERT CELL TYPES. 

            # Iterate through the peaks and lables file and obtain 
            # scores for each peak.
            y_true = our_peaks_and_labels.labels
            y_pred = []
            y_scores = []
            for pk in our_peaks_and_labels.peaks:
                predicted_labels, predicted_scores = \
                label_and_score_peak_with_chipseq_peaks([output_bed_file],pk)
                y_pred.append(predicted_labels[0])
                y_scores.append(float(predicted_scores[0]))

            # Score the results
            result = ClassificationResult(y_true, np.array(y_pred), np.array(y_scores))
            print(result)

if __name__ == '__main__':
    main()
