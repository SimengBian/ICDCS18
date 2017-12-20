class Counter:
    def __init__(self, val=0):
        self.val = val

    def get_and_inc(self):
        tmp_val = self.val
        self.val += 1
        return tmp_val


def get_delta_from_spr(succ_pr):
    from scipy.stats import norm
    return 1/(2 * float(norm.ppf( (1+succ_pr) / 2 )))

# if __name__ == '__main__':
#     succ_prs = [i*0.1 for i in range(1, 11)]
#
#     for pr in succ_prs:
#         delta = get_delta_from_spr(pr)
#         print('delta({:.2}) leads to succ. pr. {:.1%}'.format(delta, pr))
