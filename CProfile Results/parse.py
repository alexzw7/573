import pstats

p = pstats.Stats('results_stock.pstats')
p.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(10)
