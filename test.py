import hmm
import string, time
import numpy as np


if __name__ == '__main__':
  corpus = ['corpus.low']

  alpha = string.ascii_lowercase
  space = ' '

  vocab = alpha + space
  dic = dict(zip(vocab, range(len(vocab))))
  inv_dic = {v:k for k,v in dic.items()}

  O = []
  for f in corpus:
    with open(f, 'r') as fp:
      s = fp.read().strip()
    O.append(np.asarray([dic[_] for _ in s if _ in dic], dtype='int32'))

  n_A = 2
  n_B = len(dic)

  A = np.random.rand(n_A,n_A)
  A /= A.sum(axis=1)[:,np.newaxis]
  B = np.random.rand(n_A,n_B)
  B /= B.sum(axis=1)[:,np.newaxis]
  start = np.random.rand(n_A)
  start /= start.sum()

  np.set_printoptions(precision=5, suppress=True)
############################################################

  begin = time.time()
  print('begin =', begin)
  hmm.baum_welch(A, B, start, O, 500, eps=1e-20, verbose=True)
  print('cost =', time.time() - begin)

  print(A)
  print(start)

  print('emission prob:')  
  for i in range(n_B):
    print(inv_dic[i], B[:,i])
