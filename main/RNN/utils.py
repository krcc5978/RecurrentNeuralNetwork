import numpy as np


def get_csv_data(csv_path):
    with open(csv_path, 'r') as f:
        csv_data = f.read()

    # csvデータの分解
    record_data = csv_data.split('\n')
    new_record_data = []
    for record in record_data:
        record = record.split(',')
        new_record = [float(r.replace('"', '')) for r in record[1:]]
        new_record_data.append(new_record[1:])

    return new_record_data


def make_sequence_data(data, timesteps):
    sequence_data=[]
    teacher_data=[]
    for i in range(len(data) - timesteps - 1):
        sequence = [data[i + t] for t in range(timesteps)]
        sequence_data.append(sequence)
        teacher_data.append(data[i + timesteps + 1])

    return sequence_data, teacher_data


def make_teacher_data(base_path, teacher_directory_list, timesteps):

    for teacher_directory, counter in zip(teacher_directory_list, range(len(teacher_directory_list))):
        # csvデータ読み込み
        csv_path = base_path + teacher_directory + '.csv'
        new_record = get_csv_data(csv_path)

        train_sequence_data, teacher_secuence_data = make_sequence_data(new_record, timesteps)

    return np.array(train_sequence_data), np.array(teacher_secuence_data)
