"""
Address Class from SPHINCS

Code heavily inspired by https://github.com/tottifi/sphincs-python/tree/master
"""

class ADDR:
    # Types
    WOTS_HASH = 0
    WOTS_PK = 1
    TREE = 2
    FORS_TREE = 3
    FORS_ROOTS = 4

    def __init__(self):
        self.layer = 0
        self.tree_addr = 0

        self.type = 0

        # Words that define role depending on type
        self.word1 = 0
        self.word2 = 0
        self.word3 = 0

    def copy(self):
        addr = ADDR()
        addr.layer = self.layer
        addr.tree_addr = self.tree_addr

        addr.type = self.type
        addr.word1 = self.word1
        addr.word2 = self.word2
        addr.word3 = self.word3

        return addr
    
    def to_binary(self):
        addr = self.layer.to_bytes(4, byteorder='big')
        addr += self.tree_addr.to_bytes(12, byteorder='big')

        addr += self.type.to_bytes(4, byteorder='big')
        addr += self.word1.to_bytes(4, byteorder='big')
        addr += self.word2.to_bytes(4, byteorder='big')
        addr += self.word3.to_bytes(4, byteorder='big')

        return addr
    
    def reset_words(self):
        self.word1 = 0
        self.word2 = 0
        self.word3 = 0


    # GETTERS / SETTERS

    def set_type(self, type: int):
        self.type = type

        self.word1 = 0
        self.word2 = 0
        self.word3 = 0

    def set_layer_addr(self, layer: int):
        self.layer = layer

    def set_tree_addr(self, tree_addr: int):
        self.tree_addr = tree_addr

    def set_key_pair_addr(self, kp_addr: int):
        self.word1 = kp_addr

    def get_key_pair_addr(self):
        return self.word1
    
    def set_chain_addr(self, chain_addr: int):
        self.word2 = chain_addr

    def set_tree_height(self, height: int):
        self.word2 = height

    def get_tree_height(self):
        return self.word2
    
    def set_hash_addr(self, hash_addr: int):
        self.word3 = hash_addr

    def set_tree_idx(self, tree_idx: int):
        self.word3 = tree_idx

    def get_tree_idx(self):
        return self.word3