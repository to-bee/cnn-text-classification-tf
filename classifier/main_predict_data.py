import os

import django

os.environ['DJANGO_SETTINGS_MODULE'] = 'millionaire.settings'
django.setup()

import numpy as np

from ml.predictor import ClassifierPredictor
from ml.pb import Pb


def predict_data(props, rows):
    predictor = ClassifierPredictor(props=props, pb=Pb(props=props))
    probs = predictor.fetch_probs(x_batch=np.matrix(rows))
    for prob_row in probs:
        prediction_idx = np.argmax(prob_row)
        prediction = props.id2char(prediction_idx)
        prob_row_str = ', '.join(['{0:1.3f}'.format(prob) for prob in prob_row])
        print(f'Predicted label: {prediction}, Probabilities for (lost,won,draw): {prob_row_str}')


if __name__ == "__main__":
    for props in [LinearRegressionClassifierInformation(), WinOrLoseCnnClassifierInformation(), WinOrLoseLogisticRegressionClassifierInformation()]:
        rows = []
        rows.append(props.create_row(
            team_1_normalized_world_ranking_nr=1,
            team_2_normalized_world_ranking_nr=100,
        ))

        rows.append(props.create_row(
            team_1_normalized_world_ranking_nr=100,
            team_2_normalized_world_ranking_nr=100,
        ))

        rows.append(props.create_row(
            team_1_normalized_world_ranking_nr=100,
            team_2_normalized_world_ranking_nr=1,
        ))
        predict_data(props=props, rows=rows)
        print()
