from itertools import product

def main():
    for i, entry in enumerate(product('ACGT', repeat=5)):
        print ">%s" % "".join(entry)
        print "".join(entry)
    return

main()
