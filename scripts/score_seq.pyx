cdef code_base(char base):
    if base == 'A':
        return 0
    if base == 'a':
        return 0
    if base == 'C':
        return 1
    if base == 'c':
        return 1
    if base == 'G':
        return 2
    if base == 'g':
        return 2
    if base == 'T':
        return 3
    if base == 't':
        return 3
    return -1

def score_seq_from_pwm(char* seq, pwm):
    print pwm
    print [code_base(base) for base in seq]
    return 0
