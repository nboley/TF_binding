import os, sys

conn = psycopg2.connect("host=mitra dbname=cisbp user=nboley")

def load_all_motifs():
    cur = conn.cursor()    
    query = "select tf_id, motif_id, tf_name, tf_species, pwm from related_motifs_mv NATURAL JOIN pwms where tf_species = 'Homo_sapiens' and rank = 1;"
    cur.execute(query)
    for res in cur.fetchall():
        print res
        break

def main():
    motifs = load_all_motifs()

main()
