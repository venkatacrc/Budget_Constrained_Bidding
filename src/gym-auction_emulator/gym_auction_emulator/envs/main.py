

import gflags

"""
Dataset column indexes
TODO: replace dummy indexes with real indexes
"""

FLAGS = gflags.FLAGS

flags.DEFINE_integer('market_price', 10, 'mkt price index inside the bid reqs')
flags.DEFINE_integer('click', 11, 'click index inside the bid reqs')
flags.DEFINE_integer('conversion', 12, 'conversion index inside the bid reqs')