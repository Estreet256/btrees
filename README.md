# btrees

def median_split(x, columns, min_data):
    if len(columns) == 0 or x.shape[0] < min_data:
        return None
    x_col = x[:, columns[0]]
    median = np.median(x_col)
    rows_left = x_col <= median
    result = {
        'column': columns[0],
        'threshold': median,
        'left': median_split(x[rows_left], columns[1:], min_data),
        'right': median_split(x[~rows_left], columns[1:], min_data)
    }
    return result

def process_splits(node, index=0):
    left, right = node['left'], node['right']
    if left is None or right is None:
        return {index: (node['column'], node['threshold'], -1, -1)}, index, index + 1
    root_index, index = index, index + 1
    left_result, left_first, index = process_splits(left, index)
    right_result, right_first, index = process_splits(right, index)
    result = left_result
    result.update(right_result)
    result[root_index] = node['column'], node['threshold'], left_first, right_first
    return result, root_index, index

def splits_arrays(split_dict):
    length = max(split_dict.keys()) + 1
    columns = np.zeros(length, dtype=np.int32)
    thresholds = np.zeros(length, dtype=np.float32)
    wheretos = np.zeros((length, 2), dtype=np.int32)
    for index, (column, threshold, left, right) in split_dict.items():
        columns[index] = column
        thresholds[index] = threshold
        wheretos[index, :] = left, right
    return columns, thresholds, wheretos
   
@njit(parallel=True)
def leaf_indices(x, leaf_index_array, columns, thresholds, wheretos):
    for n in prange(x.shape[0]):
        index = 0
        while True:
            column, threshold, left, right = columns[index], thresholds[index], wheretos[index, 0], wheretos[index, 1]
            if left < 0:
                leaf_index_array[n] = index
                break
            value = x[n, column]
            go_left = False
            if value <= threshold:
                go_left = True
            index = left if go_left else right
           
def leaf_stats(y, leaf_indices, leaf_counts, leaf_sums):
    for n in prange(y.shape[0]):
        index = leaf_indices[n]
        leaf_counts[index] += 1
        leaf_sums[index] += y[n]
           
def fit_tree(x, y, splits):
    processed, _, _ = process_splits(splits)
    columns, thresholds, wheretos = splits_arrays(processed)
    num_nodes = wheretos.max() + 1
    leaf_index_array = np.zeros(x.shape[0], dtype=np.int32)
    leaf_counts_array = np.zeros(num_nodes, dtype=np.int32)
    leaf_sums_array = np.zeros(num_nodes, dtype=np.float64)
    leaf_indices(x, leaf_index_array, columns, thresholds, wheretos)
    leaf_stats(y, leaf_index_array, leaf_counts_array, leaf_sums_array)
    return processed, leaf_counts_array, leaf_sums_array

def predict_tree(x, processed, leaf_means):
    columns, thresholds, wheretos = splits_arrays(processed)
    leaf_index_array = np.zeros(x.shape[0], dtype=np.int32)
    leaf_indices(x, leaf_index_array, columns, thresholds, wheretos)
    return leaf_means[leaf_index_array]    
