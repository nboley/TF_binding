import psycopg2

from pyTFbindtools.ENCODE_ChIPseq_tools import (
    find_ENCODE_chipseq_experiment_ids,
    find_ENCODE_DCC_peaks,
    find_or_insert_experiment_from_called_peaks
)

from pyTFbindtools.DB import (
    encode_chipseq_exp_is_in_db, 
    conn, 
    load_genome_metadata_from_name,
    encode_ChIPseq_peak_file_is_in_db,
    sync_ENCODE_chipseq_peak_files
)

def insert_chipseq_experiment_into_db(exp_id, annotation):
    """Download and insert ENCODE experiment metadata.
    """
    #if encode_chipseq_exp_is_in_db(exp_id):
    #    return
    
    called_peaks = list(find_ENCODE_DCC_peaks(exp_id))
    if len(called_peaks) == 0: return
    
    # insert the experiment and target into the DB if necessary
    num_inserted = find_or_insert_experiment_from_called_peaks(called_peaks)
    cur = conn.cursor()
    for called_peak in called_peaks:
        if called_peak.assembly != annotation.name: 
            continue
        if encode_ChIPseq_peak_file_is_in_db(called_peak.file_loc):
            continue
        # add the peak data
        query = """
        INSERT INTO encode_chipseq_peak_files
        (
            encode_experiment_id, 
            bsid, 
            rep_key, 
            file_format, 
            file_format_type, 
            file_output_type, 
            remote_filename,
            annotation
        )
        VALUES 
        (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        try: 
            cur.execute(query, [
                called_peak.exp_id, called_peak.bsid, called_peak.rep_key,
                called_peak.file_format,
                called_peak.file_format_type,
                called_peak.output_type,
                'http://encodeproject.org' + called_peak.file_loc,
                annotation.id])
        except psycopg2.IntegrityError, inst:
            print( "ERROR", inst )
    conn.commit()
    return

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(
        description='Sync ENCODE ChIP-seq experiments in the DB.')

    parser.add_argument( '--assembly',  required=True, choices=['hg19', 'mm9'],
        help='Which genome and assembly to target.')

    parser.add_argument(
        '--process-ChIP-seq', action='store_true', default=False,
        help='Add new chipseq experiments.')

    parser.add_argument(
        '--process-DNASE', action='store_true', default=False,
        help='Add new DNASE experiments.')

    args = parser.parse_args()

    return args

def main():
    args = parse_arguments()
    annotation = load_genome_metadata_from_name(args.assembly)
    if args.process_ChIP_seq:
        for exp_id in find_ENCODE_chipseq_experiment_ids(annotation):
            insert_chipseq_experiment_into_db(exp_id, annotation)
        sync_ENCODE_chipseq_peak_files()

    if args.process_DNASE:
        raise NotImplemented, "DNASE isn't implemented yet (we're waiting on Daniel"

    return


if __name__ == '__main__':
    main()
