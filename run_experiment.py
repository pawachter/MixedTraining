import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.config.optimizer.set_jit(True)

import keras
keras.utils.disable_interactive_logging()

from data.data_manager import DataManager
from models.model_builder import ModelBuilder
from models.model_trainer import ModelTrainer
from utils.results_saver import ResultsSaver
import gc


class ExperimentRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_manager = DataManager(cfg.data, cfg.training.batch_size)
        self.model_builder = ModelBuilder(cfg.models, cfg.data, cfg.compile)
        self.results_saver = ResultsSaver(cfg.exp_dir)
        self.model_trainer = ModelTrainer(cfg.training, self.results_saver)

    def run_experiment(self):
        """
        **Run the experiment.**

        """

        # The validation and test set stay the same throughout all experiments
        val_data = self.data_manager.get_validation_data()
        test_data = self.data_manager.get_test_data()

        # Train each model in each setting once
        for setting in self.cfg.training.settings:
            if setting == "simple_mixed":
                for proportion in range(11):

                    train_data = self.data_manager.get_training_data(setting, self.cfg.seed, proportion)
                    for architecture in list(vars(self.cfg.models)):
                        keras.backend.clear_session()
                        model = self.model_builder.create_model(architecture)
                        model.summary()
                        model = self.model_trainer.train_simple_mixed(model, train_data, val_data, setting,
                                                                      architecture, proportion)
                        self.results_saver.save_test_results(
                            model=model,
                            test_dataset=test_data,
                            setting=setting,
                            architecture=architecture,
                            proportion=proportion
                        )
                        del model
                    del train_data

            elif setting == "fine-tuned":
                for proportion in range(1,10):
                    pretrain_data, fine_tune_data = self.data_manager.get_training_data(setting, self.cfg.seed,
                                                                                        proportion)

                    for architecture in list(vars(self.cfg.models)):
                        keras.backend.clear_session()
                        model = self.model_builder.create_model(architecture)
                        model = self.model_trainer.train_with_fine_tuning(model, pretrain_data, fine_tune_data,
                                                                          val_data, setting, architecture, proportion)
                        self.results_saver.save_test_results(
                            model=model,
                            test_dataset=test_data,
                            setting=setting,
                            architecture=architecture,
                            proportion=proportion,
                            training_phase=setting
                        )
                        del model
                    del pretrain_data
                    del fine_tune_data
