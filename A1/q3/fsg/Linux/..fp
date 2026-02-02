# *******************************************************
# fsg 1.37 (PAFI 1.0) Copyright 2003, Regents of the University of Minnesota
# 
# Transaction File Information --------------------------
#   Transaction File Name:                       ../Datasets/GraphTranFile
#   Number of Input Transactions:                2
#   Number of Distinct Edge Labels:              1
#   Number of Distinct Vertex Labels:            2
#   Average Number of Edges In a Transaction:    4
#   Average Number of Vertices In a Transaction: 4
#   Max Number of Edges In a Transaction:        6
#   Max Number of Vertices In a Transaction:     4
# 
# Options -----------------------------------------------
#   Min Output Pattern Size:                     1
#   Max Output Pattern Size:                     2147483647(INT_MAX)
#   Min Support Threshold:                       5.0% (1 transaction)
#   Generate Only Maximal Patterns:              No
#   Generate PC-List:                            No
#   Generate TID-List:                           No
# 
# Outputs -----------------------------------------------
#   Frequent Pattern File:                       ..fp
# 
t # 1-0, 1
v 0 V
v 1 W
u 0 1 E
t # 1-1, 2
v 0 V
v 1 V
u 0 1 E
t # 2-0, 1
v 0 V
v 1 V
v 2 W
u 0 1 E
u 0 2 E
t # 2-1, 2
v 0 V
v 1 V
v 2 V
u 0 1 E
u 0 2 E
t # 2-2, 1
v 0 W
v 1 V
v 2 V
u 0 1 E
u 0 2 E
t # 3-0, 1
v 0 V
v 1 V
v 2 W
u 0 1 E
u 0 2 E
u 1 2 E
t # 3-1, 2
v 0 V
v 1 V
v 2 V
u 0 1 E
u 0 2 E
u 1 2 E
t # 3-2, 1
v 0 W
v 1 V
v 2 V
v 3 V
u 0 1 E
u 0 2 E
u 0 3 E
t # 3-3, 1
v 0 V
v 1 V
v 2 V
v 3 W
u 0 1 E
u 0 2 E
u 0 3 E
t # 3-4, 1
v 0 V
v 1 W
v 2 V
v 3 V
u 0 1 E
u 0 2 E
u 1 3 E
t # 3-5, 1
v 0 V
v 1 V
v 2 V
v 3 W
u 0 1 E
u 0 2 E
u 1 3 E
t # 4-0, 1
v 0 W
v 1 V
v 2 V
v 3 V
u 0 1 E
u 0 2 E
u 0 3 E
u 1 2 E
t # 4-1, 1
v 0 V
v 1 V
v 2 V
v 3 W
u 0 1 E
u 0 2 E
u 0 3 E
u 1 2 E
t # 4-2, 1
v 0 V
v 1 V
v 2 W
v 3 V
u 0 1 E
u 0 2 E
u 0 3 E
u 1 2 E
t # 4-3, 1
v 0 V
v 1 V
v 2 V
v 3 W
u 0 1 E
u 0 2 E
u 1 3 E
u 2 3 E
t # 5-0, 1
v 0 V
v 1 V
v 2 V
v 3 W
u 0 1 E
u 0 2 E
u 0 3 E
u 1 2 E
u 1 3 E
t # 5-1, 1
v 0 V
v 1 W
v 2 V
v 3 V
u 0 1 E
u 0 2 E
u 0 3 E
u 1 2 E
u 1 3 E
t # 6-0, 1
v 0 V
v 1 V
v 2 V
v 3 W
u 0 1 E
u 0 2 E
u 0 3 E
u 1 2 E
u 1 3 E
u 2 3 E
#   Size Candidates Frequent Patterns 
#   1               2                 
#   2               3                 
#   3    9          6                 
#   4    10         4                 
#   5    2          2                 
#   6    1          1                 
# 
#   Largest Frequent Pattern Size:               6
#   Total Number of Candidates Generated:        22
#   Total Number of Frequent Patterns Found:     18
# 
# Timing Information ------------------------------------
#   Elapsed User CPU Time:                       0.0[sec]
# *******************************************************
