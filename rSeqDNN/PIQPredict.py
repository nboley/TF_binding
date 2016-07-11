import csv
import sys

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

def create_piq_peaks_file(score_csv_file, output_peak_file):
    with open(score_csv_file, "r") as f_handle:
        score_csv_reader = csv.reader(f_handle)

        with open(output_peak_file,"w+") as output_file:
            peak_csv_writer = csv.writer(output_file, quoting=csv.QUOTE_ALL)

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

                # Write new row to output file
                peak_csv_writer.writerow(output_row)

def main():
    print(sys.argv[1])
    print(sys.argv[2])
    create_piq_peaks_file(sys.argv[1], sys.argv[2])

if __name__ == '__main__':
    main()



