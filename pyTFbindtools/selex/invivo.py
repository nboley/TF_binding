import re

import numpy as np

from pyDNAbinding.misc import optional_gzip_open, load_fastq
from pyDNAbinding.binding_model import FixedLengthDNASequences

def upsample(seqs, num_seqs):
    new_seqs = []
    new_seqs.extend(seqs)
    while len(new_seqs) < num_seqs:
        new_seqs.extend(seqs[:num_seqs-len(new_seqs)])
    return new_seqs

class SelexDBConn(object):
    def __init__(self, host, dbname, user, exp_id):
        import psycopg2
        self.conn = psycopg2.connect(
            "host=%s dbname=%s user=%s" % (host, dbname, user))
        self.exp_id = exp_id
        return

    def get_factor_name(self):
        cur = self.conn.cursor()
        query = "SELECT tf_name FROM selex_experiments NATURAL JOIN tfs WHERE selex_exp_id = '%s';"
        cur.execute(query, (self.exp_id,))
        res = cur.fetchall()
        return res[0][0]

    def get_factor_id(self):
        cur = self.conn.cursor()
        query = "SELECT tf_id FROM selex_experiments WHERE selex_exp_id = '%s';"
        cur.execute(query, (self.exp_id,))
        res = cur.fetchall()
        return res[0][0]

    def insert_model_into_db(
            self, mo, validation_lhd):
        assert isinstance(mo, EnergeticDNABindingModel)
        motif_len = mo.motif_len
        cur = self.conn.cursor()
        query = """
        INSERT INTO new_selex_models
          (selex_exp_id, motif_len, consensus_energy, ddg_array, validation_lhd)
          VALUES 
          (%s, %s, %s, %s, %s) 
        RETURNING key
        """
        cur.execute(query, (
            self.exp_id, 
            motif_len, 
            float(mo.ref_energy), 
            mo.ddg_array.tolist(),
            float(validation_lhd)
        ))
        self.conn.commit()
        return

    def get_primer_and_fnames(self):
        cur = self.conn.cursor()
        query = """
        SELECT rnd, primer, fname 
          FROM selex_round
         WHERE selex_exp_id = '%i'
         ORDER BY rnd;
        """
        cur.execute(query % self.exp_id)
        primers = set()
        fnames = {}
        for rnd, primer, fname in cur.fetchall():
            fnames[int(rnd)] = fname
            primers.add(primer)

        assert len(primers) == 1
        primer = list(primers)[0]
        query = """
        SELECT fname 
          FROM selex_background_sequences
         WHERE primer = '%s';
        """
        cur.execute(query % primer)
        res = cur.fetchall()
        if len(res) == 0:
            bg_fname = None
        else:
            bg_fname = res[0][0]
        
        return primer, fnames, bg_fname

    def get_fnames(self):
        return self.get_primer_and_fnames()[1:]

    def get_dna_and_prot_conc(self):
        cur = self.conn.cursor()
        query = """
        SELECT dna_conc, prot_conc 
          FROM selex_round
         WHERE selex_exp_id = '%i';
        """
        cur.execute(query % self.exp_id)
        concs = set(cur.fetchall())
        assert len(concs) == 1
        return concs.pop()

def load_sequences(fnames, max_num_seqs_per_file=float('inf')):
    fnames = list(fnames)
    rnds_and_seqs = {}
    rnd_nums = [int(x.split("_")[-1].split(".")[0]) for x in fnames]
    rnds_and_fnames = dict(zip(rnd_nums, fnames))
    for rnd, fname in rnds_and_fnames.iteritems():
        with optional_gzip_open(fname) as fp:
            loader = load_fastq if ".fastq" in fname else load_text_file
            rnds_and_seqs[rnd] = loader(fp, max_num_seqs_per_file)
    return rnds_and_seqs

def load_sequence_data(selex_db_conn,
                       seq_fps,
                       background_seq_fp, 
                       max_num_seqs_per_file, 
                       min_num_background_sequences):
    if selex_db_conn is not None:
        assert seq_fps is None and background_seq_fp is None
        fnames, bg_fname = selex_db_conn.get_fnames()
        seq_fps = [ optional_gzip_open(fname) for fname in fnames.values() ]
        background_seq_fp = optional_gzip_open(bg_fname)

    pyTFbindtools.log("Loading sequences", 'VERBOSE')
    rnds_and_seqs = load_sequences(
        (x.name for x in seq_fps), max_num_seqs_per_file)

    """
    if  max_num_seqs_per_file < min_num_background_sequences:
        pyTFbindtools.log(
            "WARNING: reducing the number of background sequences to --max-num-seqs-per-file ")
        min_num_background_sequences = max_num_seqs_per_file
    else:
        min_num_background_sequences = min_num_background_sequences
    """
    min_num_background_sequences = max(
        min_num_background_sequences, 
        max(len(seqs) for seqs in rnds_and_seqs.values())
    )
    background_seqs = None
    if background_seq_fp is not None:
        with optional_gzip_open(background_seq_fp.name) as fp:
            background_seqs = load_fastq(fp, max_num_seqs_per_file)
    else:
        background_seqs = sample_random_seqs(
            min_num_background_sequences, 
            len(rnds_and_seqs.values()[0][0]))

    if len(background_seqs) < min_num_background_sequences:
        pyTFbindtools.log(
            "Too few (%i) background sequences were provided - sampling an additional %i uniform random sequences" % (
                len(background_seqs), 
                min_num_background_sequences-len(background_seqs)
            ), "VERBOSE")
        seq_len = len(rnds_and_seqs.values()[0][0])
        assert len(background_seqs) == 0 or len(background_seqs[0]) == seq_len,\
            "Background sequence length does not match sequence length."
        background_seqs.extend( 
            sample_random_seqs(
                min_num_background_sequences-len(background_seqs), seq_len)
        )

    return rnds_and_seqs, background_seqs

class SelexExperiment():
    @property
    def id(self):
        selex_exp_id = self.selex_exp_id
        factor_name = self.factor_name
        factor_id = self.factor_id
        return 'invitro_%s_%s_%s' % (
            factor_name, factor_id, selex_exp_id)

    @property
    def seq_length(self):
        return self.fwd_seqs.shape[3]

    def get_coded_seqs_and_labels(self, unbnd_seqs, bnd_seqs, primer):
        left_flank, right_flank = re.split('[0-9]{1,4}N', primer)
        left_flank = left_flank.rjust(self.seq_pad).replace(' ', 'N')
        right_flank = right_flank.ljust(self.seq_pad).replace(' ', 'N')

        num_seqs = min(len(unbnd_seqs), len(bnd_seqs))
        unbnd_seqs = FixedLengthDNASequences(
            [left_flank + x + right_flank for x in unbnd_seqs[:num_seqs]], 
            include_shape=False)
        bnd_seqs = FixedLengthDNASequences(
            [left_flank + x + right_flank for x in bnd_seqs[:num_seqs]], 
            include_shape=False)

        permutation = np.random.permutation(2*num_seqs)
        fwd_one_hot_seqs = np.vstack(
            (unbnd_seqs.fwd_one_hot_coded_seqs, bnd_seqs.fwd_one_hot_coded_seqs)
        )[permutation,:,:]
        fwd_one_hot_seqs = np.swapaxes(fwd_one_hot_seqs, 1, 2)[:,None,:,:]

        if unbnd_seqs.fwd_shape_features is None:
            shape_seqs = None
        else:
            fwd_shape_seqs = np.vstack(
                (unbnd_seqs.fwd_shape_features, bnd_seqs.fwd_shape_features)
            )[permutation,:,:]
            rc_shape_seqs = np.vstack(
                (unbnd_seqs.rc_shape_features, bnd_seqs.rc_shape_features)
            )[permutation,:,:]
            shape_seqs = np.concatenate(
                (fwd_shape_seqs, rc_shape_seqs), axis=2)[:,None,:,:]

        labels = np.hstack(
            (np.zeros(len(unbnd_seqs), dtype='float32'), 
             np.ones(len(bnd_seqs), dtype='float32'))
        )[permutation]
        
        return fwd_one_hot_seqs, shape_seqs, labels # rc_one_hot_seqs, 

    def __init__(
            self, selex_exp_id, n_samples, validation_split=0.1, seq_pad=40):
        self.selex_exp_id = selex_exp_id
        self.n_samples = n_samples
        self.seq_pad = seq_pad
        self.validation_split = validation_split

        # load connect to the DB, and find the factor name
        selex_db_conn = SelexDBConn(
            'mitra', 'cisbp', 'nboley', selex_exp_id)
        self.factor_name = selex_db_conn.get_factor_name()
        self.factor_id = selex_db_conn.get_factor_id()

        # load the sequencess
        print "Loading sequences for %s-%s (exp ID %i)" % (
            self.factor_name, self.factor_id, self.selex_exp_id)
        self.primer, fnames, bg_fname = selex_db_conn.get_primer_and_fnames()
        if bg_fname is None:
            raise NoBackgroundSequences(
                "No background sequences for %s (%i)." % (
                    self.factor_name, self.selex_exp_id))

        rnd_num = max(fnames.keys())
        with optional_gzip_open(fnames[rnd_num]) as fp:
            bnd_seqs = load_fastq(fp, self.n_samples)
        with optional_gzip_open(bg_fname) as fp:
            unbnd_seqs = load_fastq(fp, self.n_samples)
        if len(bnd_seqs[0]) < 20:
            raise SeqsTooShort("Seqs too short for %s (exp %i)." % (
                self.factor_name, selex_exp_id))
        
        if self.n_samples is not None:
            if len(unbnd_seqs) < self.n_samples:
                unbnd_seqs = upsample(unbnd_seqs, self.n_samples)
            if len(bnd_seqs) < self.n_samples:
                bnd_seqs = upsample(bnd_seqs, self.n_samples)

        (self.fwd_seqs, self.shape_seqs, self.labels 
             ) = self.get_coded_seqs_and_labels(
                 unbnd_seqs, bnd_seqs, self.primer)
        self.n_samples = self.fwd_seqs.shape[0]
        self._training_index = int(self.n_samples*self.validation_split)
        self.train_fwd_seqs = self.fwd_seqs[self._training_index:]
        self.train_labels = self.labels[self._training_index:]
        
        self.validation_fwd_seqs = self.fwd_seqs[:self._training_index]
        self.validation_labels = self.labels[:self._training_index]
        
        return

    def iter_batches(
            self, batch_size, data_subset, repeat_forever, **kwargs):
        if data_subset == 'train': 
            fwd_seqs = self.train_fwd_seqs
            labels = self.train_labels
        elif data_subset == 'validation':
            fwd_seqs = self.validation_fwd_seqs
            labels = self.validation_labels
        else:
            raise ValueError, "Unrecognized data_subset type '%s'" % data_subset

        i = 0
        n = fwd_seqs.shape[0]//batch_size
        if n <= 0: raise ValueError, "Maximum batch size is %i (requested %i)" \
           % (fwd_seqs.shape[0], batch_size)
        while repeat_forever is True or i < n:
            # yield a subset of the data
            subset = slice((i%n)*batch_size, (i%n+1)*batch_size)
            yield {'fwd_seqs': fwd_seqs[subset], 'output': labels[subset,None]}
            i += 1
        
        return
    
    def iter_train_data(self, batch_size, repeat_forever=False, **kwargs):
        return self.iter_batches(batch_size, 'train', repeat_forever)

    def iter_validation_data(self, batch_size, repeat_forever=False, **kwargs):
        return self.iter_batches(batch_size, 'validation', repeat_forever)

class SelexData():
    def load_tfs_grpd_by_family(self):
        raise NotImplementedError, "This fucntion just exists to save teh sql query"
        query = """
          SELECT tfs.family_id, family_name, array_agg(tfs.factor_name) 
            FROM selex_experiments NATURAL JOIN tfs NATURAL JOIN tf_families 
        GROUP BY family_id, family_name;
        """ # group tfs by their families

        pass

    @staticmethod
    def find_all_selex_experiments():
        import psycopg2
        conn = psycopg2.connect(
            "host=%s dbname=%s user=%s" % ('mitra', 'cisbp', 'nboley'))
        cur = conn.cursor()    
        query = """
          SELECT selex_experiments.tf_id, tfs.tf_name, array_agg(distinct selex_exp_id)
            FROM selex_experiments NATURAL JOIN tfs NATURAL JOIN roadmap_matched_chipseq_peaks
        GROUP BY selex_experiments.tf_id, tfs.tf_name
        ORDER BY tf_name;
        """
        cur.execute(query)
        # filter out the TFs that dont have background sequence or arent long enough
        res = [ x for x in cur.fetchall() 
                if x[1] not in ('E2F1', 'E2F4', 'POU2F2', 'PRDM1', 'RXRA')
                #and x[1] in ('MAX',) # 'YY1', 'RFX5', 'USF', 'PU1', 'CTCF')
        ]
        return res

    def add_selex_experiment(self, selex_exp_id):
        self.experiments.append(
            SelexExperiment(
                selex_exp_id, self.n_samples, self.validation_split)
        )

    def add_all_selex_experiments_for_factor(self, factor_name):
        exps_added = 0
        for tf_id, i_factor_name, selex_exp_ids in self.find_all_selex_experiments():
            if i_factor_name == factor_name:
                for exp_id in selex_exp_ids:
                    self.add_selex_experiment(exp_id)
                    exps_added += 1
        if exps_added == 0:
            print "Couldnt add for", factor_name
        #assert exps_added > 0
        return

    def add_all_selex_experiments(self):
        for tf_id, factor_name, selex_exp_ids in self.find_all_selex_experiments():
            for selex_exp_id in selex_exp_ids:
                try: 
                    self.add_selex_experiment(selex_exp_id)
                except SeqsTooShort, inst:
                    print inst
                    continue
                except NoBackgroundSequences, inst:
                    print inst
                    continue
                except Exception, inst:
                    raise
    
    @property
    def factor_names(self):
        return [exp.factor_name for exp in self]

    def __iter__(self):
        for experiment in self.experiments:
            yield experiment
        return

    def __init__(self, n_samples, validation_split=0.1):
        self.n_samples = n_samples
        self.validation_split = 0.1
        self.experiments = []

    def __len__(self):
        return len(self.experiments)
