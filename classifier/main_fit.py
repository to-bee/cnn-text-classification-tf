from ml.pb import Pb
from ml.plate_cnn.plate_ci import PlateClassifierInformation
from ml.plate_cnn.plate_cnn_df import PlateDataFrame
from ml.plate_cnn.plate_cnn_trainer import PlateCnnClassifierTrainer


def create_datasets(name, path, props, restore=False, create_variants=False):
    df = PlateDataFrame(props=props, name=name, restore=restore)
    if not restore:
        df.read_data(path=path, create_variants=create_variants)
        df.save_dataset()
        print(f'Dataset {name} created')

    df.show_summary()
    return df


if __name__ == '__main__':
    classifier_information = PlateClassifierInformation()

    restore_df = False
    path = classifier_information.resource_train_path
    df_train = create_datasets(name='train', path=path, props=classifier_information, restore=restore_df, create_variants=True)

    path = classifier_information.resource_test_path
    df_test = create_datasets(name='test', path=path, props=classifier_information, restore=restore_df, create_variants=False)

    pb = Pb(props=classifier_information)
    cnn = PlateCnnClassifierTrainer(props=classifier_information, pb=pb, restore_model=False)
    (global_step, ckpt, final_test_accuracy, final_test_cost) = cnn.fit(df_train=df_train, df_test=df_test)
    pb.save(ckpt, final_test_accuracy, final_test_cost)
