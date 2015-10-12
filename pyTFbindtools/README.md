Code Snippets
=============

<b>Accessing chromatin accessible regions with TF peak labels</b>

The below will load an object that contains all roadmap chromatin accessible regions and CTCF binding labels (1 for bound, 0 for not). ctcf_peaks_and_labels is a PeaksAndLabels object and it has a couple useful functions. There's a caching layer on top that prevent the peaks from being labelled multiple times.  


    # import the function
    from pyTFbindtools.peaks import load_chromatin_accessible_peaks_and_chipseq_labels_from_DB 
    # load all chromatin accessible peaks that have a matching CTCF sample.
    ctcf_peaks_and_labels = load_chromatin_accessible_peaks_and_chipseq_labels_from_DB(
      # use CTCF (this ID can be found in the cisbp db on mitra by running
      # select * from tfs where tf_name = 'CTCF' and tf_species = 'Homo_sapiens';
      tf_id='T044268_1.02', 
      # center all chromatin accessible region peaks on their summit, and make the 
      # peak 1000 bp wide (500 bp in either direction of the summit)
      half_peak_width=500,
      # only use 5000 peaks from each sample (set to None for all peaks)
      max_n_peaks_per_sample=5000,
      # if set, skip regions that are negatives in the IDR optimal peak set,
      # but positives in the relaxed set
      # THE CURRENT IMPLEMENTATION IS POOR BECAUSE WE DONT YET HAVE A STANDARDIZED 
      # RELAXED PEAK SET
      skip_ambiguous_peaks=False
    )


