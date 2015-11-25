import sys

from pyTFbindtools.motif_tools import (
    build_model_scores_hdf5_file, 
    score_regions_from_hdf5, 
    load_selex_models_from_db,
    aggregate_region_scores)

from grit.lib.multiprocessing_utils import (
    ThreadSafeFile, run_in_parallel )

def build_hdf5():
    try:
        build_model_scores_h5_file(int(sys.argv[1]), sys.argv[2])
    except IOError:
        pass
    else:
        print sys.argv[2]

def load_peaks(fname):
    rv = []
    with open(fname) as fp:
        for line in fp:
            data = line.split()
            rv.append((data[0], int(data[1]), int(data[2])))
    return rv

def score_regions_worker(ofp, model, annotation_id, regions):
    print "Scoring regions for:", model.motif_id
    try:
        for i, (region, scores) in enumerate(score_regions_from_hdf5(
                regions, annotation_id, model)):
            agg_scores = aggregate_region_scores(scores)
            ofp.write("\t".join([str(x) for x in region]
                                + [model.tf_id, str(annotation_id)]
                                + ["%.5e" % x for x in agg_scores]) 
                      + "\n")
            if i%10000 == 0:
                print i, len(regions), region, agg_scores
    except IOError:
        print "Skipping motif."
    return
    
def main():
    regions = load_peaks(sys.argv[1])
    models = load_selex_models_from_db()
    annotation_id = 1
    
    ofp = ThreadSafeFile('output.txt', "w")
    all_args = [(ofp, model, annotation_id, regions) for model in models]
    run_in_parallel(8, score_regions_worker, all_args)
    ofp.close()

if __name__ == '__main__':
    main()
