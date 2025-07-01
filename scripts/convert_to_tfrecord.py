import numpy as np
import tensorflow as tf
import os

def write_tfrecord(data_path, output_path, split_name):
    data = np.load(data_path)
    X, y = data['X'], data['y']
    filename = os.path.join(output_path, f"{split_name}.tfrecords")
    
    with tf.io.TFRecordWriter(filename) as writer:
        for i in range(len(X)):
            image = X[i].tobytes()
            label = y[i].tobytes()
            
            feature = {
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=X[i].shape))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_npz', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--split_name', type=str, required=True)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    write_tfrecord(args.input_npz, args.output_dir, args.split_name)
