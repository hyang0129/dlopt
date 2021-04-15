from .. import optimization as op
from .. import ea as ea
from .. import nn as nn
from .. import sampling as sp
from .. import util as ut
from . import problems as pr
from abc import ABC, abstractmethod
from keras.layers.recurrent import LSTM
import numpy as np
import time
import gc
from keras import backend as K
from dlopt.optimization import Solution
from pathlib import Path
import os
import keras
import numpy as np
import json

class TimeSeriesHybridMRSProblem(pr.TimeSeriesMAERandSampProblem):
    """ Mean Absolute Error Random Sampling RNN Problem
    """
    def __init__(self,
                 dataset,
                 targets,
                 verbose=0,
                 num_samples=30,
                 min_layers=1,
                 max_layers=1,
                 min_neurons=1,
                 max_neurons=1,
                 min_look_back=1,
                 max_look_back=1,
                 sampler=sp.MAERandomSampling,
                 nn_builder_class=nn.RNNBuilder,
                 nn_trainer_class=nn.TrainGradientBased,
                 dropout=0.5,
                 epochs=10,
                 batch_size=5,
                 **kwargs):
        super().__init__(dataset,
                         targets,
                         verbose,
                         num_samples,
                         min_layers,
                         max_layers,
                         min_neurons,
                         max_neurons,
                         min_look_back,
                         max_look_back,
                         sampler,
                         nn_builder_class,
                         **kwargs)
        self.nn_trainer_class = nn_trainer_class
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size

    def solution_as_result(self,
                           solution):
        solution_desc = {}
        model, layers, look_back = self.decode_solution(solution)
        solution_desc['layers'] = layers
        solution_desc['look_back'] = look_back
        solution_desc['fitness'] = solution.fitness
        model, nn_metric, pred = self._train(model,
                                             look_back,
                                             self.dropout,
                                             self.epochs)
        solution_desc['testing_metrics'] = nn_metric
        solution_desc['y_predicted'] = pred.tolist()
        solution_desc['config'] = str(model.get_config())
        return model, solution_desc

    def _train(self,
               model,
               look_back,
               dropout,
               epochs):
        K.clear_session()
        if self.verbose > 1:
            print('Session cleared')
        model = model.__class__.from_config(model.get_config())
        start = time.time()
        trainer = self.nn_trainer_class(verbose=self.verbose,
                                        **self.kwargs)
        model = self.builder_class.add_dropout(model,
                                               dropout)
        self.builder_class.init_weights(model,
                                        ut.random_uniform,
                                        low=-0.5,
                                        high=0.5)
        trainer.load_from_model(model)
        self.dataset.training_data.look_back = look_back
        self.dataset.training_data.batch_size = self.batch_size
        self.dataset.validation_data.look_back = look_back
        self.dataset.validation_data.batch_size = self.batch_size
        self.dataset.testing_data.look_back = look_back
        self.dataset.testing_data.batch_size = self.batch_size
        trainer.train(self.dataset.training_data,
                      validation_dataset=self.dataset.validation_data,
                      epochs=epochs,
                      **self.kwargs)
        metrics, pred = trainer.evaluate(self.dataset.testing_data,
                                         **self.kwargs)
        evaluation_time = time.time() - start
        metrics['evaluation_time'] = evaluation_time
        del trainer
        gc_out = gc.collect()
        if self.verbose:
            print(metrics)
        return model, metrics, pred


class SelfAdjMuPLambdaUniform(ea.EABase):
    """ (Mu+Lambda) basic algorithm
    """
    def __init__(self,
                 problem,
                 seed=None,
                 verbose=0):
        super().__init__(problem,
                         seed,
                         verbose)
        # We add the default parameter values
        self.params.update({'p_mutation_i': 0.1,
                            'p_mutation_e': 0.1,
                            'mutation_max_step': 2})
        self.last_avgs = {}

    def mutate(self,
               solution):
        ea.uniformMutation(solution.get_encoded('architecture'),
                           self.params['p_mutation_i'],
                           self.params['mutation_max_step'])
        ea.uniformLengthMutation(solution.get_encoded('architecture'),
                                 self.params['p_mutation_e'])

    def select(self,
               population):
        return ea.binaryTournament(population)

    def replace(self,
                population,
                offspring):
        return ea.elitistPlusReplacement(population,
                                         offspring)

    def call_on_generation(self,
                           population):
        super().call_on_generation(population)
        avgs = {}
        for target in population[0].targets:
            avgs[target] = np.mean([sol.get_fitness(target)
                                    for sol in population])

        print("Mutation parameters before tuning",
              self.params['p_mutation_i'],
              self.params['p_mutation_e'],
              self.params['mutation_max_step'])
        if len(self.last_avgs) > 0:
            diffs = []
            for target in population[0].targets:
                diff = avgs[target] - self.last_avgs[target]
                if ((population[0].targets[target] < 0 and diff <= 0) or
                        (population[0].targets[target] > 0 and diff <= 0)):
                    diffs.append(1)
                else:
                    diffs.append(-1)
            if np.sum(diffs) > 0:
                # We are improving (on average)
                self.params['p_mutation_i'] = min(1.0, self.params['p_mutation_i'] * 1.5)
                self.params['p_mutation_e'] = min(1.0, self.params['p_mutation_e'] * 1.5)
                self.params['mutation_max_step'] = self.params['mutation_max_step'] * 1.5
            else:
                self.params['p_mutation_i'] = max(10e-10, self.params['p_mutation_i'] / 4)
                self.params['p_mutation_e'] = max(10e-10, self.params['p_mutation_e'] / 4)
                self.params['mutation_max_step'] = max(1, self.params['mutation_max_step'] / 4)
            # 
        self.last_avgs = avgs
        # gc_out = gc.collect()

        # print("GC collect", gc_out)
        print("Averages:", str(avgs))
        print("Mutation parameters after tuning",
              self.params['p_mutation_i'],
              self.params['p_mutation_e'],
              self.params['mutation_max_step'])


class StaticSelfAdjMuPLambdaUniform(SelfAdjMuPLambdaUniform):
    def call_on_generation(self, population):
        print('keeping mutation params the same')

class TimeSeriesTrainProblem(op.Problem):
    """ Optimize an RNN architecture based on the results
    of pre-trained networks
    """
    def __init__(self,
                 dataset,
                 targets,
                 verbose=0,
                 min_layers=1,
                 max_layers=1,
                 min_neurons=1,
                 max_neurons=1,
                 min_look_back=1,
                 max_look_back=1,
                 train_epochs=10,
                 test_epochs=10,
                 nn_builder_class=nn.RNNBuilder,
                 nn_trainer_class=nn.TrainGradientBased,
                 dropout=0.5,
                 **kwargs):
        super().__init__(dataset,
                         targets,
                         verbose,
                         **kwargs)
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.min_neurons = min_neurons
        self.max_neurons = max_neurons
        self.min_look_back = min_look_back
        self.max_look_back = max_look_back
        self.dropout = dropout
        self.train_epochs = train_epochs
        self.test_epochs = test_epochs
        self.nn_trainer_class = nn_trainer_class
        self.builder_class = nn_builder_class

    def evaluate(self,
                 solution):
        if solution.is_evaluated():
            if self.verbose > 1:
                print('Solution already evaluated')
            return
        model, layers, look_back = self.decode_solution(solution)
        model, results, _ = self._train(model,
                                        look_back,
                                        self.dropout,
                                        self.train_epochs)
        del model
        gc.collect()
        if self.verbose > 1:
            print({'layers': layers,
                   'look_back': look_back,
                   'results': results})
        for target in self.targets:
            solution.set_fitness(target,
                                 results[target])

    def next_solution(self):
        solution = op.Solution(self.targets,
                               ['architecture'])
        num_layers = np.random.randint(low=self.min_layers,
                                       high=(self.max_layers + 1))
        layers = np.random.randint(low=self.min_neurons,
                                   high=(self.max_neurons + 1),
                                   size=num_layers)
        look_back = np.random.randint(low=self.min_look_back,
                                      high=(self.max_look_back + 1),
                                      size=1)
        solution.set_encoded('architecture',
                             np.concatenate((look_back, layers)).tolist())
        return solution

    def validate_solution(self,
                          solution):
        encoded = solution.get_encoded('architecture')
        # look back
        if len(encoded) < 1:
            encoded.append(self.min_look_back)
        if encoded[0] < self.min_look_back:
            encoded[0] = self.min_look_back
        elif encoded[0] > self.max_look_back:
            encoded[0] = self.max_look_back
        elif not isinstance(encoded[0], int):
            encoded[0] = int(encoded[0])
        # layers
        while (len(encoded) - 1) < self.min_layers:
            encoded.append(self.min_neurons)
        while (len(encoded) - 1) > self.max_layers:
            encoded.pop()
        for i in range(1, len(encoded)):
            if encoded[i] > self.max_neurons:
                encoded[i] = self.max_neurons
            elif encoded[i] < self.min_neurons:
                encoded[i] = self.min_neurons
            elif not isinstance(encoded[i], int):
                encoded[i] = int(encoded[i])

    def decode_solution(self,
                        solution):
        layers = ([self.dataset.input_dim] +
                  solution.get_encoded('architecture')[1:] +
                  [self.dataset.output_dim])
        look_back = solution.get_encoded('architecture')[0]
        model = self.builder_class.build_model(layers,
                                               verbose=self.verbose,
                                               **self.kwargs)
        return model, layers, look_back

    def solution_as_result(self,
                           solution):
        solution_desc = {}
        model, layers, look_back = self.decode_solution(solution)
        solution_desc['layers'] = layers
        solution_desc['look_back'] = look_back
        solution_desc['fitness'] = solution.fitness
        model, metrics, pred = self._train(model,
                                           look_back,
                                           self.dropout,
                                           self.test_epochs)
        solution_desc['testing_metrics'] = metrics
        solution_desc['y_predicted'] = pred.tolist()
        solution_desc['config'] = str(model.get_config())
        return model, solution_desc

    def _train(self,
               model,
               look_back,
               dropout,
               epochs):
        K.clear_session()
        if self.verbose > 1:
            print('Session cleared')
        model = model.__class__.from_config(model.get_config())
        start = time.time()
        trainer = self.nn_trainer_class(verbose=self.verbose,
                                        **self.kwargs)
        model = self.builder_class.add_dropout(model,
                                               dropout)
        self.builder_class.init_weights(model,
                                        ut.random_uniform,
                                        low=-0.5,
                                        high=0.5)
        trainer.load_from_model(model)
        self.dataset.training_data.look_back = look_back
        self.dataset.validation_data.look_back = look_back
        self.dataset.testing_data.look_back = look_back
        trainer.train(self.dataset.training_data,
                      validation_dataset=self.dataset.validation_data,
                      epochs=epochs,
                      **self.kwargs)
        metrics, pred = trainer.evaluate(self.dataset.testing_data,
                                         **self.kwargs)
        evaluation_time = time.time() - start
        metrics['evaluation_time'] = evaluation_time
        del trainer
        gc_out = gc.collect()
        if self.verbose > 1:
            print("GC collect", gc_out)
            print(gc.garbage)
        if self.verbose:
            print(metrics)

        if epochs > 10:
            with open('/content/drive/Shareddrives/KAGGLE/MRS/backprop_results.txt', 'a') as f:
                metrics['total_time_seconds'] = time.time() - self.start
                metrics['total_time_minutes'] = metrics['total_time_seconds'] / 60
                f.write(json.JSONEncoder().encode(metrics))
                f.write('\n')

        return model, metrics, pred

class LamarckianTimeSeriesTrainProblem(TimeSeriesTrainProblem):

    def evaluate(self,
                 solution : Solution):

        K.clear_session()

        if solution.is_evaluated():
            if self.verbose > 1:
                print('Solution already evaluated')
            return
        model, layers, look_back = self.decode_solution(solution)

        dir = 'tmp'

        init_model = True
        if hasattr(solution, 'parent_id'):
            print('inheriting weights from solution %i to solution %i' % (solution.parent_id, solution.id))
            parent_model = self.get_parent_model(dir, solution.parent_id)
            model = self.inherit_weights(model, parent_model)
            init_model = False


        model, results, _ = self._train(model,
                                        look_back,
                                        self.dropout,
                                        self.train_epochs,
                                        init_model)

        if self.verbose > 1:
            print({'layers': layers,
                   'look_back': look_back,
                   'results': results})
        for target in self.targets:
            solution.set_fitness(target,
                                 results[target])

        Path(dir).mkdir(parents=True, exist_ok = True)
        solution.save(dir, model)

        del model
        gc.collect()

    def _train(self,
               model,
               look_back,
               dropout,
               epochs,
               init_model=True):

        if self.verbose > 1:
            print('Session cleared')

        # model = model.__class__.from_config(model.get_config())

        start = time.time()
        trainer = self.nn_trainer_class(verbose=self.verbose,
                                        **self.kwargs)
        model = self.builder_class.add_dropout(model,
                                               dropout)

        if init_model:
            print("initialization model using random uniform")
            self.builder_class.init_weights(model,
                                            ut.random_uniform,
                                            low=-0.5,
                                            high=0.5)

        trainer.load_from_model(model)
        self.dataset.training_data.look_back = look_back
        self.dataset.validation_data.look_back = look_back
        self.dataset.testing_data.look_back = look_back
        trainer.train(self.dataset.training_data,
                      validation_dataset=self.dataset.validation_data,
                      epochs=epochs,
                      **self.kwargs)
        metrics, pred = trainer.evaluate(self.dataset.testing_data,
                                         **self.kwargs)
        evaluation_time = time.time() - start
        metrics['evaluation_time'] = evaluation_time
        del trainer
        gc_out = gc.collect()

        if self.verbose > 1:
            print("GC collect", gc_out)
            print(gc.garbage)

        if epochs > 10:
            with open('lamarckian_results.txt', 'a') as f:
                metrics['total_time_seconds'] = time.time() - self.start
                metrics['total_time_minutes'] = metrics['total_time_seconds'] / 60
                f.write(json.JSONEncoder().encode(metrics))
                f.write('\n')

            print(metrics)

        return model, metrics, pred

    @staticmethod
    def get_parent_model(path, id):
        return keras.models.load_model(os.path.join(path, f'model_{id:03d}.h5'))

    @staticmethod
    def inherit_weights(model, parent_model):

        child = model
        parent = parent_model

        wts = child.get_weights()
        p_wts = parent.get_weights()

        flat_wts = np.concatenate([wt.reshape(-1) for wt in p_wts])

        mean = np.mean(flat_wts)
        std = np.std(flat_wts)

        inherited_weights = 0
        total_weights = 0
        for i, w in enumerate(wts):

            if i < len(p_wts):
                num_wts_in_layer = np.product(w.shape)
                total_weights += num_wts_in_layer
                # print(w.shape, p_wts[i].shape)
                if w.shape == p_wts[i].shape:
                    # print('inherited')
                    wts[i] = p_wts[i]
                    inherited_weights += num_wts_in_layer
                else:
                    # print('initialized via mean and std of parent')
                    # wts[i] = np.random.uniform(loc=mean, scale=std, size=wts[i].shape)
                    wts[i] = np.random.uniform(low=-0.5, high=0.5, size=wts[i].shape)
                    pass

            else:
                # print('more layers in child than parent initialized via mean and std of parent')
                # wts[i] = np.random.normal(loc=mean, scale=std, size=wts[i].shape)
                wts[i] = np.random.uniform(low=-0.5, high=0.5, size=wts[i].shape)
                pass

        print(f"inherited {inherited_weights * 100/ total_weights:.1f}% of {total_weights} weights" )
        model.set_weights(wts)

        return model
