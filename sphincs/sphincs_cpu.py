"""
Main SPHINC CPU Implementation

Code heavily inspired by https://github.com/tottifi/sphincs-python/tree/master
"""

import os
import math
import random
import hashlib

from addr import ADDR

def sha256(seed: bytes, addr: ADDR, data: bytes, digest_size: int) -> bytes:
    """
    Hashes the data using SHA256

    Parameters:
        seed (bytes): the seed
        addr (ADDR):  the ADDR object
        data (bytes): the data to hash
        digest_size (int):  the number of bytes to return from the hash object
    Return:
        (bytes):    the hashed data in raw binary
    """

    hasher = hashlib.sha256()

    hasher.update(seed)
    hasher.update(addr.to_binary())
    hasher.update(data)

    hash_out = hasher.digest()[:digest_size]

    return hash_out


def prf(sec_seed: bytes, addr: ADDR, digest_size: int) -> bytes:
    """
    Performs a pseudorandom function on the secret seed

    Parameters:
        sec_seed (bytes):   the secret seed
        addr (ADDR):        the ADDR object
        digest_size (int):  the number of bytes to return from the object

    Return:
        (bytes):    the output of the pseudorandom function
    """

    random.seed(int.from_bytes(sec_seed + addr.to_binary(), 'big'))
    return random.randint(0, 256 ** digest_size - 1).to_bytes(digest_size, byteorder='big')


def hash_message(r, pub_seed, pub_root, data, digest_size):
    """
    
    """

    hasher = hashlib.sha256()

    hasher.update(r)
    hasher.update(pub_seed)
    hasher.update(pub_root)
    hasher.update(data)

    hash_out = hasher.digest()[:digest_size]

    i = 0
    while len(hash_out) < digest_size:
        i += 1
        hasher = hashlib.sha256()

        hasher.update(r)
        hasher.update(pub_seed)
        hasher.update(pub_root)
        hasher.update(data)
        hasher.update(bytes([i]))

        hash_out += hasher.digest()[:digest_size - len(hash_out)]

    return hash_out

def prf_message(sec_seed, opt, m, digest_size):
    """
    
    """

    random.seed(int.from_bytes(sec_seed + opt + hash_message(b'0', b'0', b'0', m, digest_size * 2), 'big'))
    return random.randint(0, 256 ** digest_size - 1).to_bytes(digest_size, byteorder='big')


def print_bytes_bit(data):
    """
    
    """

    arr = []
    for d in data:
        for i in range(7, -1, -1):
            arr.append((d >> i) % 2)
    print(arr)


def base_w(x, w, out_len):
    """
    
    """

    v_in = 0
    v_out = 0
    total = 0
    num_bits = 0
    base_w = []

    for _ in range(0, out_len):
        if num_bits == 0:
            total = x[v_in]
            v_in += 1
            num_bits += 8
        num_bits -= math.floor(math.log(w, 2))
        base_w.append((total >> num_bits) % 2)
        v_out += 1

    return base_w


class SPHINCS():
    """
    The SPHINCS Class
    """

    def __init__(self):
        self._randomize = True

        # Defaults for SPHINCS+ 128s
        self._n = 16
        self._h = 64
        self._d = 8
        self._a = 15
        self._k = 10
        self._w = 16


        self._len1 = math.ceil(8 * self._n / math.log(self._w, 2))
        self._len2 = math.floor(math.log(self._len1 * (self._w - 1), 2) / math.log(self._w, 2)) + 1
        self._len0 = self._len1 + self._len2
        self._h_p = self._h // self._d
        self._t = 2 ** self._a

    def calc_vars(self):
        self._len1 = math.ceil(8 * self._n / math.log(self._w, 2))
        self._len2 = math.floor(math.log(self._len1 * (self._w - 1), 2) / math.log(self._w, 2)) + 1
        self._len0 = self._len1 + self._len2
        self._h_p = self._h // self._d
        self._t = 2 ** self._a

    def keygen(self):
        """
        Generates the public key (pk) and private/secret key (sk)

        Return:
            (bytes, bytes):   the public key and private key
        """

        sk, pk = self.sphincs_keygen()
        sk0, pk0 = bytes(), bytes()

        for i in sk:
            sk0 += i
        for i in pk:
            pk0 += i

        return sk0, pk0


    def sign(self, msg, sk) -> bytes:
        """
        Sign the message with the SPHINCS scheme

        Parameters:
            msg (bytes):    the message to be signed
            sk (bytes):     the secret key
        Return:
            (bytes):    the signature of the message
        """

        sk_tab = []

        for i in range(0, 4):
            sk_tab.append(sk[(i * self._n):((i + 1) * self._n)])

        sig_tab = self.sphincs_sign(msg, sk_tab)

        sig = sig_tab[0]
        for i in sig_tab[1]:    # FORS
            sig += i
        for i in sig_tab[2]:    # Hypertree
            sig += i

        return sig


    def verify(self, msg, sig, pk):
        """
        Verify the signature against the message and public key

        Parameters:
            msg (bytes):    the signed message
            sig (bytes):    the signature of the message
            pk (bytes):     the public key
        Return:
            (bool): True if the signature is correct, False otherwise
        """

        pk_tab = []

        for i in range(0, 2):
            pk_tab.append(pk[(i * self._n):((i + 1) * self._n)])

        sig_tab = []

        sig_tab += [sig[:self._n]]

        sig_tab += [[]] # FORS
        for i in range(self._n, self._n + self._k * (self._a + 1) * self._n, self._n):
            sig_tab[1].append(sig[i:(i + self._n)])

        sig_tab += [[]] # Hypertree

        for i in range(self._n + self._k * (self._a + 1) * self._n,
                       self._n + self._k * (self._a + 1) * self._n + (self._h + self._d * self._len0) * self._n,
                       self._n):
            sig_tab[2].append(sig[i:(i + self._n)])

        return self.sphincs_verify(msg, sig_tab, pk_tab)


    # GETTERS / SETTERS

    def set_security(self, sec):
        self._n = sec
        self.calc_vars()

    def set_n(self, n):
        self._n = n
        self.calc_vars()

    def get_security(self):
        return self._n

    def set_winternitz(self, w):
        if w == 4 or w == 16 or w == 256:
            self._w = w
        self.calc_vars()

    def set_w(self, w):
        if w == 4 or w == 16 or w == 256:
            self._w = w
        self.calc_vars()

    def get_winternitz(self):
        return self._w

    def set_hypertree_height(self, h):
        self._h = h
        self.calc_vars()

    def set_h(self, h):
        self._h = h
        self.calc_vars()

    def get_hypertree_height(self):
        return self._h

    def set_hypertree_layers(self, d):
        self._d = d
        self.calc_vars()

    def set_d(self, d):
        self._d = d
        self.calc_vars()

    def get_hypertree_layers(self):
        return self._d

    def set_fors_trees_number(self, k):
        self._k = k
        self.calc_vars()

    def set_k(self, k):
        self._k = k
        self.calc_vars()

    def get_fors_trees_number(self):
        return self._k

    def set_fors_trees_height(self, a):
        self._a = a
        self.calc_vars()

    def set_a(self, a):
        self._a = a
        self.calc_vars()

    def get_fors_trees_height(self):
        return self._a


    # UTILS

    def sig_wots_from_sig_xmss(self, sig):
        return sig[0:self._len0]

    def auth_from_sig_xmss(self, sig):
        return sig[self._len0:]

    def sigs_xmss_from_sig_ht(self, sig):
        sigs = []
        for i in range(0, self._d):
            sigs.append(sig[i * (self._h_p + self._len0):(i + 1) * (self._h_p + self._len0)])

        return sigs

    def auths_from_sig_fors(self, sig):
        sigs = []
        for i in range(0, self._k):
            sigs.append([])
            sigs[i].append(sig[(self._a + 1) * i])
            sigs[i].append(sig[((self._a + 1) * i + 1):((self._a + 1) * (i + 1))])

        return sigs

    # WOTS+

    # Input: Input string X, start index i, number of steps s, public seed PK.seed, address ADRS
    # Output: value of F iterated s times on X
    def chain(self, x, i, s, pub_seed, addr: ADDR):
        """
        
        """

        if s == 0:
            return bytes(x)

        if (i + s) > (self._w - 1):
            return -1

        tmp = self.chain(x, i, s - 1, pub_seed, addr)

        addr.set_hash_addr(i + s - 1)
        tmp = sha256(pub_seed, addr, tmp, self._n)

        return tmp

    # # Input: secret seed SK.seed, address ADRS
    # # Output: WOTS+ private key sk
    # def wots_sk_gen(self, sec_seed, addr: ADDR):  # Not necessary
    #     """
        
    #     """

    #     sk = []
    #     for i in range(0, self._len0):
    #         addr.set_chain_addr(i)
    #         addr.set_hash_addr(0)
    #         sk.append(prf(sec_seed, addr.copy(), self._n))
    #     return sk

    # Input: secret seed SK.seed, address ADRS, public seed PK.seed
    # Output: WOTS+ public key pk
    def wots_keygen(self, sec_seed, pub_seed, addr: ADDR):
        wots_pk_addr = addr.copy()
        tmp = bytes()
        for i in range(0, self._len0):
            addr.set_chain_addr(i)
            addr.set_hash_addr(0)
            sk = prf(sec_seed, addr.copy(), self._n)
            tmp += bytes(self.chain(sk, 0, self._w - 1, pub_seed, addr.copy()))

        wots_pk_addr.set_type(ADDR.WOTS_PK)
        wots_pk_addr.set_key_pair_addr(addr.get_key_pair_addr())

        pk = sha256(pub_seed, wots_pk_addr, tmp, self._n)

        sk = []
        for i in range(0, self._len0):
            addr.set_chain_addr(i)
            addr.set_hash_addr(0)
            sk.append(prf(sec_seed, addr.copy(), self._n))

        return sk, pk

    # Input: Message M, secret seed SK.seed, public seed PK.seed, address ADRS
    # Output: WOTS+ signature sig
    def wots_sign(self, m, sec_seed, pub_seed, addr: ADDR):
        csum = 0

        msg = base_w(m, self._w, self._len1)

        for i in range(0, self._len1):
            csum += self._w - 1 - msg[i]

        padding = (self._len2 * math.floor(math.log(self._w, 2))) % 8 if (self._len2 * math.floor(math.log(self._w, 2))) % 8 != 0 else 8
        csum = csum << (8 - padding)
        csumb = csum.to_bytes(math.ceil((self._len2 * math.floor(math.log(self._w, 2))) / 8), byteorder='big')
        csumw = base_w(csumb, self._w, self._len2)
        msg += csumw

        sig = []
        for i in range(0, self._len0):
            addr.set_chain_addr(i)
            addr.set_hash_addr(0)
            sk = prf(sec_seed, addr.copy(), self._n)
            sig += [self.chain(sk, 0, msg[i], pub_seed, addr.copy())]

        return sig

    def wots_pk_from_sig(self, sig, m, pub_seed, addr: ADDR):
        csum = 0
        wots_pk_addr = addr.copy()

        msg = base_w(m, self._w, self._len1)

        for i in range(0, self._len1):
            csum += self._w - 1 - msg[i]

        padding = (self._len2 * math.floor(math.log(self._w, 2))) % 8 if (self._len2 * math.floor(math.log(self._w, 2))) % 8 != 0 else 8
        csum = csum << (8 - padding)
        csumb = csum.to_bytes(math.ceil((self._len2 * math.floor(math.log(self._w, 2))) / 8), byteorder='big')
        csumw = base_w(csumb, self._w, self._len2)
        msg += csumw

        tmp = bytes()
        for i in range(0, self._len0):
            addr.set_chain_addr(i)
            tmp += self.chain(sig[i], msg[i], self._w - 1 - msg[i], pub_seed, addr.copy())

        wots_pk_addr.set_type(ADDR.WOTS_PK)
        wots_pk_addr.set_key_pair_addr(addr.get_key_pair_addr())
        pk_sig = sha256(pub_seed, wots_pk_addr, tmp, self._n)
        return pk_sig

    # XMSS
    # =================================================

    # Input: Secret seed SK.seed, start index s, target node height z, public seed PK.seed, address ADRS
    # Output: n-byte root node - top node on Stack
    def treehash(self, sec_seed, s, z, pub_seed, addr: ADDR):
        if s % (1 << z) != 0:
            return -1

        stack = []

        for i in range(0, 2 ** z):
            addr.set_type(ADDR.WOTS_HASH)
            addr.set_key_pair_addr(s + i)
            _, node = self.wots_keygen(sec_seed, pub_seed, addr.copy())

            addr.set_type(ADDR.TREE)
            addr.set_tree_height(1)
            addr.set_tree_idx(s + i)

            if len(stack) > 0:
                while stack[len(stack) - 1]['height'] == addr.get_tree_height():
                    addr.set_tree_idx((addr.get_tree_idx() - 1) // 2)
                    node = sha256(pub_seed, addr.copy(), stack.pop()['node'] + node, self._n)
                    addr.set_tree_height(addr.get_tree_height() + 1)

                    if len(stack) <= 0:
                        break

            stack.append({'node': node, 'height': addr.get_tree_height()})

        return stack.pop()['node']

    # Input: Secret seed SK.seed, public seed PK.seed, address ADRS
    # Output: XMSS public key PK
    def xmss_pk_gen(self, sec_seed, pub_key, addr: ADDR):
        pk = self.treehash(sec_seed, 0, self._h_p, pub_key, addr.copy())
        return pk

    # Input: n-byte message M, secret seed SK.seed, index idx, public seed PK.seed, address ADRS
    # Output: XMSS signature SIG_XMSS = (sig || AUTH)
    def xmss_sign(self, m, sec_seed, idx, pub_seed, addr: ADDR):
        auth = []
        for j in range(0, self._h_p):
            ki = math.floor(idx // 2 ** j)
            if ki % 2 == 1:  # XORING idx/ 2**j with 1
                ki -= 1
            else:
                ki += 1

            auth += [self.treehash(sec_seed, ki * 2 ** j, j, pub_seed, addr.copy())]

        addr.set_type(ADDR.WOTS_HASH)
        addr.set_key_pair_addr(idx)

        sig = self.wots_sign(m, sec_seed, pub_seed, addr.copy())
        sig_xmss = sig + auth
        return sig_xmss

    # Input: index idx, XMSS signature SIG_XMSS = (sig || AUTH), n-byte message M, public seed PK.seed, address ADRS
    # Output: n-byte root value node[0]
    def xmss_pk_from_sig(self, idx, sig_xmss, m, pub_seed, addr: ADDR):
        addr.set_type(ADDR.WOTS_HASH)
        addr.set_key_pair_addr(idx)
        sig = self.sig_wots_from_sig_xmss(sig_xmss)
        auth = self.auth_from_sig_xmss(sig_xmss)

        node0 = self.wots_pk_from_sig(sig, m, pub_seed, addr.copy())
        node1 = 0

        addr.set_type(ADDR.TREE)
        addr.set_tree_idx(idx)
        for i in range(0, self._h_p):
            addr.set_tree_height(i + 1)

            if math.floor(idx / 2 ** i) % 2 == 0:
                addr.set_tree_idx(addr.get_tree_idx() // 2)
                node1 = sha256(pub_seed, addr.copy(), node0 + auth[i], self._n)
            else:
                addr.set_tree_idx((addr.get_tree_idx() - 1) // 2)
                node1 = sha256(pub_seed, addr.copy(), auth[i] + node0, self._n)

            node0 = node1

        return node0

    # HYPERTREE XMSS
    # =================================================

    # Input: Private seed SK.seed, public seed PK.seed
    # Output: HT public key PK_HT
    def ht_pk_gen(self, sec_seed, pub_seed):
        addr = ADDR()
        addr.set_layer_addr(self._d - 1)
        addr.set_tree_addr(0)
        root = self.xmss_pk_gen(sec_seed, pub_seed, addr.copy())
        return root

    # Input: Message M, private seed SK.seed, public seed PK.seed, tree index idx_tree, leaf index idx_leaf
    # Output: HT signature SIG_HT
    def ht_sign(self, m, sec_seed, pub_seed, idx_tree, idx_leaf):
        addr = ADDR()
        addr.set_layer_addr(0)
        addr.set_tree_addr(idx_tree)

        sig_tmp = self.xmss_sign(m, sec_seed, idx_leaf, pub_seed, addr.copy())
        sig_ht = sig_tmp
        root = self.xmss_pk_from_sig(idx_leaf, sig_tmp, m, pub_seed, addr.copy())

        for j in range(1, self._d):
            idx_leaf = idx_tree % 2 ** self._h_p
            idx_tree = idx_tree >> self._h_p

            addr.set_layer_addr(j)
            addr.set_tree_addr(idx_tree)

            sig_tmp = self.xmss_sign(root, sec_seed, idx_leaf, pub_seed, addr.copy())
            sig_ht = sig_ht + sig_tmp

            if j < self._d - 1:
                root = self.xmss_pk_from_sig(idx_leaf, sig_tmp, root, pub_seed, addr.copy())

        return sig_ht

    # Input: Message M, signature SIG_HT, public seed PK.seed, tree index idx_tree, leaf index idx_leaf, HT public key PK_HT
    # Output: Boolean
    def ht_verify(self, m, sig_ht, pub_seed, idx_tree, idx_leaf, pub_key_ht):
        addr = ADDR()

        sigs_xmss = self.sigs_xmss_from_sig_ht(sig_ht)
        sig_tmp = sigs_xmss[0]

        addr.set_layer_addr(0)
        addr.set_tree_addr(idx_tree)
        node = self.xmss_pk_from_sig(idx_leaf, sig_tmp, m, pub_seed, addr)

        for j in range(1, self._d):
            idx_leaf = idx_tree % 2 ** self._h_p
            idx_tree = idx_tree >> self._h_p

            sig_tmp = sigs_xmss[j]

            addr.set_layer_addr(j)
            addr.set_tree_addr(idx_tree)

            node = self.xmss_pk_from_sig(idx_leaf, sig_tmp, node, pub_seed, addr)

        if node == pub_key_ht:
            return True
        else:
            return False

    # FORS

    # Input: secret seed SK.seed, address ADRS, secret key index idx = it+j
    # Output: FORS private key sk
    def fors_sk_gen(self, sec_seed, addr: ADDR, idx):
        addr.set_tree_height(0)
        addr.set_tree_idx(idx)
        sk = prf(sec_seed, addr.copy(), self._n)

        return sk

    # Input: Secret seed SK.seed, start index s, target node height z, public seed PK.seed, address ADRS
    # Output: n-byte root node - top node on Stack
    def fors_treehash(self, sec_seed, s, z, pub_seed, addr: ADDR):
        if s % (1 << z) != 0:
            return -1

        stack = []

        for i in range(0, 2 ** z):
            addr.set_tree_height(0)
            addr.set_tree_idx(s + i)
            sk = prf(sec_seed, addr.copy(), self._n)
            node = sha256(pub_seed, addr.copy(), sk, self._n)

            addr.set_tree_height(1)
            addr.set_tree_idx(s + i)
            if len(stack) > 0:
                while stack[len(stack) - 1]['height'] == addr.get_tree_height():
                    addr.set_tree_idx((addr.get_tree_idx() - 1) // 2)
                    node = sha256(pub_seed, addr.copy(), stack.pop()['node'] + node, self._n)

                    addr.set_tree_height(addr.get_tree_height() + 1)

                    if len(stack) <= 0:
                        break
            stack.append({'node': node, 'height': addr.get_tree_height()})

        return stack.pop()['node']

    # Input: Secret seed SK.seed, public seed PK.seed, address ADRS
    # Output: FORS public key PK
    def fors_pk_gen(self, sec_seed, pub_seed, addr: ADDR):
        fors_pk_addr = addr.copy()

        root = bytes()
        for i in range(0, self._k):
            root += self.fors_treehash(sec_seed, i * self._t, self._a, pub_seed, addr)

        fors_pk_addr.set_type(ADDR.FORS_ROOTS)
        fors_pk_addr.set_key_pair_addr(addr.get_key_pair_addr())
        pk = sha256(pub_seed, fors_pk_addr, root, self._n)
        return pk

    # Input: Bit string M, secret seed SK.seed, address ADRS, public seed PK.seed
    # Output: FORS signature SIG_FORS
    def fors_sign(self, m, sec_seed, pub_seed, addr):
        m_int = int.from_bytes(m, 'big')
        sig_fors = []

        for i in range(0, self._k):
            idx = (m_int >> (self._k - 1 - i) * self._a) % self._t

            addr.set_tree_height(0)
            addr.set_tree_idx(i * self._t + idx)
            sig_fors += [prf(sec_seed, addr.copy(), self._n)]

            auth = []

            for j in range(0, self._a):
                s = math.floor(idx // 2 ** j)
                if s % 2 == 1:  # XORING idx/ 2**j with 1
                    s -= 1
                else:
                    s += 1

                auth += [self.fors_treehash(sec_seed, i * self._t + s * 2 ** j, j, pub_seed, addr.copy())]

            sig_fors += auth

        return sig_fors

    # Input: FORS signature SIG_FORS, (k lg t)-bit string M, public seed PK.seed, address ADRS
    # Output: FORS public key
    def fors_pk_from_sig(self, sig_fors, m, pub_seed, addr: ADDR):
        m_int = int.from_bytes(m, 'big')

        sigs = self.auths_from_sig_fors(sig_fors)
        root = bytes()

        for i in range(0, self._k):
            idx = (m_int >> (self._k - 1 - i) * self._a) % self._t

            sk = sigs[i][0]
            addr.set_tree_height(0)
            addr.set_tree_idx(i * self._t + idx)
            node0 = sha256(pub_seed, addr.copy(), sk, self._n)
            node1 = 0

            auth = sigs[i][1]
            addr.set_tree_idx(i * self._t + idx)  # Really Useful?

            for j in range(0, self._a):
                addr.set_tree_height(j + 1)

                if math.floor(idx / 2 ** j) % 2 == 0:
                    addr.set_tree_idx(addr.get_tree_idx() // 2)
                    node1 = sha256(pub_seed, addr.copy(), node0 + auth[j], self._n)
                else:
                    addr.set_tree_idx((addr.get_tree_idx() - 1) // 2)
                    node1 = sha256(pub_seed, addr.copy(), auth[j] + node0, self._n)

                node0 = node1

            root += node0

        fors_pk_addr = addr.copy()
        fors_pk_addr.set_type(ADDR.FORS_ROOTS)
        fors_pk_addr.set_key_pair_addr(addr.get_key_pair_addr())

        pk = sha256(pub_seed, fors_pk_addr, root, self._n)
        return pk

    # SPHINCS IMPLEMENTATION
    # =================================================

    # Input: (none)
    # Output: SPHINCS+ key pair (SK,PK)
    def sphincs_keygen(self):
        sec_seed = os.urandom(self._n)
        sec_prf = os.urandom(self._n)
        pub_seed = os.urandom(self._n)

        pub_root = self.ht_pk_gen(sec_seed, pub_seed)

        return [sec_seed, sec_prf, pub_seed, pub_root], [pub_seed, pub_root]

    # Input: Message M, private key SK = (SK.seed, SK.prf, PK.seed, PK.root)
    # Output: SPHINCS+ signature SIG
    def sphincs_sign(self, m, sk):
        addr = ADDR()

        sec_seed = sk[0]
        sec_prf = sk[1]
        pub_seed = sk[2]
        pub_root = sk[3]

        opt = bytes(self._n)
        if self._randomize:
            opt = os.urandom(self._n)
        r = prf_message(sec_prf, opt, m, self._n)
        sig = [r]

        size_md = math.floor((self._k * self._a + 7) / 8)
        size_idx_tree = math.floor((self._h - self._h // self._d + 7) / 8)
        size_idx_leaf = math.floor((self._h // self._d + 7) / 8)

        digest = hash_message(r, pub_seed, pub_root, m, size_md + size_idx_tree + size_idx_leaf)
        tmp_md = digest[:size_md]
        tmp_idx_tree = digest[size_md:(size_md + size_idx_tree)]
        tmp_idx_leaf = digest[(size_md + size_idx_tree):len(digest)]

        md_int = int.from_bytes(tmp_md, 'big') >> (len(tmp_md) * 8 - self._k * self._a)
        md = md_int.to_bytes(math.ceil(self._k * self._a / 8), 'big')

        idx_tree = int.from_bytes(tmp_idx_tree, 'big') >> (len(tmp_idx_tree) * 8 - (self._h - self._h // self._d))
        idx_leaf = int.from_bytes(tmp_idx_leaf, 'big') >> (len(tmp_idx_leaf) * 8 - (self._h // self._d))

        addr.set_layer_addr(0)
        addr.set_tree_addr(idx_tree)
        addr.set_type(ADDR.FORS_TREE)
        addr.set_key_pair_addr(idx_leaf)

        sig_fors = self.fors_sign(md, sec_seed, pub_seed, addr.copy())
        sig += [sig_fors]

        pk_fors = self.fors_pk_from_sig(sig_fors, md, pub_seed, addr.copy())

        addr.set_type(ADDR.TREE)
        sig_ht = self.ht_sign(pk_fors, sec_seed, pub_seed, idx_tree, idx_leaf)
        sig += [sig_ht]

        return sig

    # Input: Message M, signature SIG, public key PK
    # Output: Boolean
    def sphincs_verify(self, m, sig, pub_key):
        addr = ADDR()
        r = sig[0]
        sig_fors = sig[1]
        sig_ht = sig[2]

        pub_seed = pub_key[0]
        pub_root = pub_key[1]

        size_md = math.floor((self._k * self._a + 7) / 8)
        size_idx_tree = math.floor((self._h - self._h // self._d + 7) / 8)
        size_idx_leaf = math.floor((self._h // self._d + 7) / 8)

        digest = hash_message(r, pub_seed, pub_root, m, size_md + size_idx_tree + size_idx_leaf)
        tmp_md = digest[:size_md]
        tmp_idx_tree = digest[size_md:(size_md + size_idx_tree)]
        tmp_idx_leaf = digest[(size_md + size_idx_tree):len(digest)]

        md_int = int.from_bytes(tmp_md, 'big') >> (len(tmp_md) * 8 - self._k * self._a)
        md = md_int.to_bytes(math.ceil(self._k * self._a / 8), 'big')

        idx_tree = int.from_bytes(tmp_idx_tree, 'big') >> (len(tmp_idx_tree) * 8 - (self._h - self._h // self._d))
        idx_leaf = int.from_bytes(tmp_idx_leaf, 'big') >> (len(tmp_idx_leaf) * 8 - (self._h // self._d))

        addr.set_layer_addr(0)
        addr.set_tree_addr(idx_tree)
        addr.set_type(ADDR.FORS_TREE)
        addr.set_key_pair_addr(idx_leaf)

        pk_fors = self.fors_pk_from_sig(sig_fors, md, pub_seed, addr)

        addr.set_type(ADDR.TREE)
        return self.ht_verify(pk_fors, sig_ht, pub_seed, idx_tree, idx_leaf, pub_root)

