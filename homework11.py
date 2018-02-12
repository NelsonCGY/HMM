############################################################
# CIS 521: Final Project / Homework 11
############################################################

student_name = "Gongyao Chen"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import math
import pickle

############################################################
# Section 1: Hidden Markov Models
############################################################


def load_corpus(path):
    in_file = open(path, 'r')
    albet = 'abcdefghijklmnopqrstuvwxyz '
    s = ''
    for line in in_file:
        s += line.lower() + ' '
    in_file.close()
    result = []
    for ch in s:
        if ch in albet:
            result.append(ch)
    s = ''.join(result)
    result = s.split()
    return ' '.join(result)


def load_parameters(path):
    in_file = open(path, 'rb')
    result = pickle.load(in_file)
    init = result[0]
    trans = result[1]
    emiss = result[2]
    in_file.close()
    for key in init.keys():
        init[key] = math.log(init[key])
    for key in trans.keys():
        for one_key in trans[key].keys():
            trans[key][one_key] = math.log(trans[key][one_key])
    for key in emiss.keys():
        for one_key in emiss[key].keys():
            emiss[key][one_key] = math.log(emiss[key][one_key])
    return tuple([init, trans, emiss])


def log_sum_exp(xn):
    a = max(xn)
    tmp = 0
    for x in xn:
        tmp += math.exp(x - a)
    return a + math.log(tmp)


class HMM(object):

    def __init__(self, probabilities):
        self.init = probabilities[0]
        self.trans = probabilities[1]
        self.emiss = probabilities[2]

    def get_parameters(self):
        exp_init = {}
        exp_trans = {}
        exp_emiss = {}
        for key in self.init.keys():
            exp_init[key] = math.exp(self.init[key])
        for key in self.trans.keys():
            one_trans = {}
            for one_key in self.trans[key].keys():
                one_trans[one_key] = math.exp(self.trans[key][one_key])
            exp_trans[key] = one_trans
        for key in self.emiss.keys():
            one_emiss = {}
            for one_key in self.emiss[key].keys():
                one_emiss[one_key] = math.exp(self.emiss[key][one_key])
            exp_emiss[key] = one_emiss
        return tuple([exp_init, exp_trans, exp_emiss])

    def forward(self, sequence):
        alpha = []
        for t in xrange(len(sequence)):
            one_alpha = {}
            token = sequence[t]
            if t == 0:
                for i in xrange(len(self.init)):
                    one_alpha[i + 1] = self.init[i + 1] + self.emiss[i + 1][token]
            else:
                for i in xrange(len(self.init)):
                    xn = []
                    for j in xrange(len(self.init)):
                        xn.append(alpha[t - 1][j + 1] + self.trans[j + 1][i + 1])
                    one_alpha[i + 1] = log_sum_exp(xn) + self.emiss[i + 1][token]
            alpha.append(one_alpha)    
        return alpha

    def forward_probability(self, alpha):
        return log_sum_exp(alpha[-1].values())

    def backward(self, sequence):
        beta = []
        for x in xrange(len(sequence)):
            beta.append({})
        for t in xrange(len(sequence) - 1, -1, -1):
            if t == len(sequence) - 1:
                for i in xrange(len(self.init)):
                    beta[t][i + 1] = math.log(1)
            else:
                for i in xrange(len(self.init)):
                    xn = []
                    for j in xrange(len(self.init)):
                        xn.append(beta[t + 1][j + 1] + self.trans[i + 1][j + 1] + self.emiss[j + 1][sequence[t + 1]])
                    beta[t][i + 1] = log_sum_exp(xn)
        return beta

    def backward_probability(self, beta, sequence):
        xn = []
        for i in xrange(len(self.init)):
            xn.append(self.init[i + 1] + self.emiss[i + 1][sequence[0]] + beta[0][i + 1])
        return log_sum_exp(xn)

    def forward_backward(self, sequence):
        alpha = self.forward(sequence)
        beta = self.backward(sequence)
        gama = {}
        dic = {}
        for t in xrange(len(sequence)):
            tmp = {}
            if t == len(sequence) - 1:
                xn = []
                for i in xrange(len(self.init)):
                    xn.append(alpha[t][i + 1] + beta[t][i + 1])
                dom = log_sum_exp(xn)
                for i in xrange(len(self.init)):
                    num = alpha[t][i + 1] + beta[t][i + 1]
                    tmp[i + 1] = num - dom
            else:
                dic[t] = self.xi_matrix(t, sequence, alpha, beta)
                for i in xrange(len(self.init)):
                    tmp[i + 1] = log_sum_exp(dic[t][i + 1].values())
            gama[t] = tmp
        init = gama[0]
        
        trans = {}
        for i in xrange(len(self.init)):
            tmp = {}
            for j in xrange(len(self.init)):
                xna = []
                xnb = []
                for t in xrange(len(sequence) - 1):
                    xna.append(dic[t][i + 1][j + 1])
                    xnb.append(gama[t][i + 1])
                num = log_sum_exp(xna)
                dom = log_sum_exp(xnb)
                tmp[j + 1] = num - dom
            trans[i + 1] = tmp
            
        emiss = {}
        for (tag, words) in self.emiss.items():
            tmp = {}
            for word in words.keys():
                xna = []
                xnb = []
                for t in xrange(len(sequence)):
                    if word == sequence[t]:
                        xna.append(gama[t][tag])
                    xnb.append(gama[t][tag])
                num = log_sum_exp(xna)
                dom = log_sum_exp(xnb)
                tmp[word] = num - dom
            emiss[tag] = tmp
        return tuple([init, trans, emiss])

    def xi_matrix(self, t, sequence, alpha, beta):
        dic = {}
        xn = []
        for i in xrange(len(self.init)):
            for j in xrange(len(self.init)):
                xn.append(alpha[t][i + 1] + self.trans[i + 1][j + 1] + self.emiss[j + 1][sequence[t + 1]] + beta[t + 1][j + 1])
        dom = log_sum_exp(xn)
        for i in xrange(len(self.init)):
            tmp = {}
            for j in xrange(len(self.init)):
                num = xn[i * len(self.init) + j]
                tmp[j + 1] = num - dom
            dic[i + 1] = tmp
        return dic

    def update(self, sequence, cutoff_value):
        p_pre = self.forward_probability(self.forward(sequence))
        h = HMM(self.forward_backward(sequence))
        h_pre = h
        p_update = h.forward_probability(h.forward(sequence))
        while (p_update - p_pre > cutoff_value):
            p_pre = p_update
            h_pre = h
            h = HMM(h.forward_backward(sequence))
            p_update = h.forward_probability(h.forward(sequence))
        h_pre = h
        self.init = h_pre.init
        self.trans = h_pre.trans
        self.emiss = h_pre.emiss


############################################################
# Section 2: Feedback
############################################################


feedback_question_1 = """
10hr
"""

feedback_question_2 = """
The most challenging part is to be very careful when processing forward/backward and forward-backward part.
"""

feedback_question_3 = """
Nothing special.
"""
