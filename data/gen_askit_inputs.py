# Generate ASKIT input files from data file and exact file

import sys
import numpy as np

def output(fname, a):
    with open(fname, 'w') as f:
        for row in a:
            row_str = [str(val) if val != 0 else '0' for val in row]
            f.write(','.join(row_str))
            f.write('\n')

if __name__ == "__main__":
    ds = sys.argv[1]
    data_file = sys.argv[2]
    exact_file = sys.argv[3]

    data = None
    try: 
        data = np.loadtxt('%s' % data_file, delimiter=',')
    except Exception as e:
        print("hello")
        print(data_file)
        print("Error:", e)

    # Output space seperate file
    output('%s_askit.data' % ds, data)

    e = np.loadtxt('%s' % exact_file, delimiter=',')
    # Get query data points
    query = np.zeros((e.shape[0], data.shape[1]))
    for i in range(e.shape[0]):
        query[i] = data[int(e[i][1])]
    # Output space seperate file
    output('%s_askit_query.data' % ds, query)

