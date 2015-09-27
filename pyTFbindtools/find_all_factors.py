from predictTFPeaks import *

def main():
    motifs = load_pwms_from_db()
    for motif in motifs:
        roadmap_sample_ids = load_matching_roadmap_samples(motif.tf_name)
        if len(roadmap_sample_ids) > 0:
            print motif.tf_name, 
    return

main()
