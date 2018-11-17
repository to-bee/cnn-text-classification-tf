import os
import uuid
from typing import Tuple

from django.db.models import Avg, Min, Max, Sum

i = 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def aggregate_property(qs, aggregate_property: str):
    aggregations = qs.aggregate(Avg(aggregate_property), Min(aggregate_property), Max(aggregate_property), Sum(aggregate_property))
    return list(aggregations.values())


def normalize(current_value: int, aggregations: Tuple[float, float, float]) -> float:
    (avg_value, max_value, min_value) = aggregations
    return (current_value - avg_value) / (max_value - min_value)


def get_export_filename(name, ending):
    return '%s.%s' % (name, ending)


def list_files_of_type(path, type):
    if not os.path.isdir(path):
        raise Exception("The given path {} doesn't exist".format(path))
    if type is None:
        return [os.path.join(path, fname) for fname in os.listdir(path)]
    else:
        files = [os.path.join(path, fname) for fname in os.listdir(path)]
        return [file for file in files if file.endswith(type)]


def get_uuid():
    return str(uuid.uuid4().hex)


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def clear_folder(dir):
    if os.path.exists(dir):
        for the_file in os.listdir(dir):
            file_path = os.path.join(dir, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                else:
                    clear_folder(file_path)
                    os.rmdir(file_path)

            except Exception as e:
                print(e)
