"""
Genetic Programming in Python, with a scikit-learn inspired API.

The `gplearn.genetic` module implements Genetic Programming. These are supervised learning methods based on applying evolutionary operations on computer programs.
"""

import itertools
from abc import ABCMeta, abstractmethod
from time import time
from warnings import warn
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from utils import _partition_estimators, check_random_state
from _program import _Program
from fitness import _fitness_map, _Fitness, _extra_map
from functions import _function_map, _Function, sig1 as sigmoid
from more_functions import _extra_function_map
from collections import Counter

_all_func_dictionary = dict(_function_map,**_extra_function_map)
__all__ = ['SymbolicTransformer']
MAX_INT = np.iinfo(np.int32).max

##########################################Dynamic Mutation Prob##########################################
def _dynamic_mutation_probs(parent, method_probs):
    # np.array([self.p_crossover,self.p_subtree_mutation,self.p_hoist_mutation,self.p_point_mutation,self.p_reproduction])
    method_probs = method_probs.copy()
    if parent.depth_ > 4: # If parent is too deep, incread the probability of hoist_mutation
        method_probs[2] *= 2
    if parent.length_ > 9: # If parent is too long, increase the probability of hoist_mutation and point_mutation
        method_probs[2] *= 2
        method_probs[3] *= 2
    # Count the frequency of each variable in the parent program. If any variable appears more than 3 times, increase the probability of point_mutation
    frequency = Counter(item for item in parent.program if isinstance(item, int))
    max_count = max(frequency.values())
    if max_count > 2:
        method_probs[3] *= 2
    return method_probs

def _parallel_evolve_3D(n_programs, parents, X, y, sample_weight, seeds, params):
    """Private function used to build a batch of programs within a job."""
    n_dates, n_features, n_stocks = X.shape

    # Unpack parameters
    tournament_size = params['tournament_size']
    function_set = params['function_set']
    arities = params['arities']
    init_depth = params['init_depth']
    init_method = params['init_method']
    const_range = params['const_range']
    metric = params['_metric']
    transformer = params['_transformer']
    parsimony_coefficient = params['parsimony_coefficient']
    method_probs = params['method_probs']
    p_point_replace = params['p_point_replace']
    max_samples = params['max_samples']
    feature_names = params['feature_names']
    
    def _tournament():
        """Find the fittest individual from a sub-population."""
        contenders = random_state.randint(0, len(parents), tournament_size)
        fitness = [parents[p].fitness_ for p in contenders] # use fitness after parsimony pressure
        if metric.greater_is_better:
            parent_index = contenders[np.argmax(fitness)]
        else:
            parent_index = contenders[np.argmin(fitness)]
        return parents[parent_index], parent_index

    # Build programs
    programs = []

    for i in range(n_programs):
        random_state = check_random_state(seeds[i])
        if parents is None:
            program = None
            genome = None
        else:
            parent, parent_index = _tournament()
            # Dynamic modify mutation probabilities based on parent program
            method_probs = _dynamic_mutation_probs(parent, method_probs)
            method_probs = np.cumsum(method_probs) # calculate cumulative sum for method selection probabilities
            method = random_state.uniform() * method_probs[-1] # select method based on total probabilities (Note: method_probs may not sum to 1)
            if method < method_probs[0]:
                # crossover
                donor, donor_index = _tournament()
                program, removed, remains = parent.crossover(donor.program, random_state)
                genome = {'method': 'Crossover',
                          'parent_idx': parent_index,
                          'parent_nodes': removed,
                          'donor_idx': donor_index,
                          'donor_nodes': remains}
            elif method < method_probs[1]:
                # subtree_mutation
                program, removed, _ = parent.subtree_mutation(random_state)
                genome = {'method': 'Subtree Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': removed}
            elif method < method_probs[2]:
                # hoist_mutation
                program, removed = parent.hoist_mutation(random_state)
                genome = {'method': 'Hoist Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': removed}
            elif method < method_probs[3]:
                # point_mutation
                program, mutated = parent.point_mutation(random_state)
                genome = {'method': 'Point Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': mutated}
            else:
                # reproduction
                program = parent.reproduce()
                genome = {'method': 'Reproduction',
                          'parent_idx': parent_index,
                          'parent_nodes': []}

        program = _Program(function_set=function_set,
                           arities=arities,
                           init_depth=init_depth,
                           init_method=init_method,
                           n_features=n_features,
                           metric=metric,
                           transformer=transformer,
                           const_range=const_range,
                           p_point_replace=p_point_replace,
                           parsimony_coefficient=parsimony_coefficient,
                           feature_names=feature_names,
                           random_state=random_state,
                           program=program)

        program.parents = genome

        # Draw samples, using sample weights, and then fit
        if sample_weight is None:
            curr_sample_weight = np.ones(n_dates)
        else:
            curr_sample_weight = sample_weight.copy()
        oob_sample_weight = np.where(curr_sample_weight > 0, 0, 1)

        program.raw_fitness_ = program.raw_fitness_3D(X, y, curr_sample_weight)
        if max_samples < 1.0:
            # Calculate OOB fitness
            program.oob_fitness_ = program.raw_fitness_3D(X, y, oob_sample_weight)
        programs.append(program)
    return programs

class BaseSymbolic(BaseEstimator, metaclass=ABCMeta):
    """
    Base class for symbolic regression / classification estimators.

    Warning: This class should not be used directly.
        Use derived classes instead.

    """

    @abstractmethod
    def __init__(self,
                 *,
                 population_size=1000,
                 hall_of_fame=None,
                 n_components=None,
                 generations=20,
                 tournament_size=20,
                 stopping_criteria=0.0,
                 const_range=(-1., 1.),
                 init_depth=(2, 6),
                 init_method='half and half',
                 function_set=('add', 'sub', 'mul', 'div'),
                 transformer=None,
                 metric='mean absolute error',
                 parsimony_coefficient=0.001,
                 p_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_point_replace=0.05,
                 p_reproduction = 0.01,
                 max_samples=1.0,
                 class_weight=None,
                 feature_names=None,
                 warm_start=False,
                 low_memory=False,
                 n_jobs=1,
                 verbose=0,
                 random_state=None):

        self.population_size = population_size
        self.hall_of_fame = hall_of_fame
        self.n_components = n_components
        self.generations = generations
        self.tournament_size = tournament_size
        self.stopping_criteria = stopping_criteria
        self.const_range = const_range
        self.init_depth = init_depth
        self.init_method = init_method
        self.function_set = function_set
        self.transformer = transformer
        self.metric = metric
        self.parsimony_coefficient = parsimony_coefficient
        self.p_crossover = p_crossover
        self.p_subtree_mutation = p_subtree_mutation
        self.p_hoist_mutation = p_hoist_mutation
        self.p_point_mutation = p_point_mutation
        self.p_point_replace = p_point_replace
        self.p_reproduction = p_reproduction
        self.max_samples = max_samples
        self.class_weight = class_weight
        self.feature_names = feature_names
        self.warm_start = warm_start
        self.low_memory = low_memory
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self._satisfied_programs = []
        self._satisfied_factor_last_day_dfs = None
        self._best_programs = []

    def _verbose_reporter(self, run_details=None):
        """
        A report of the progress of the evolution process.

        Parameters
        ----------
        run_details : dict
            Information about the evolution.

        """
        if run_details is None:
            # Print header fields.
            print('     |{:^35}|{:^50}|{:^15}|'.format('Population Metrics', 'Best Individual Metrics', 'Time Remaining'))
            print('-----'+ ' ' + '-' * 35 + ' ' + '-' * 50 + ' ' + '-' * 15)
            line_format = '{:>4} {:>10} {:>10} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}'
            print(line_format.format('Gen', 'Length', 'Depth', 'Fitness', 'Length', 'Depth', 'Fitness', 'OOB_Fitness', 'Time'))

        else:
            # Estimate remaining time for run
            gen = run_details['generation'][-1]
            generation_time = run_details['generation_time'][-1]
            remaining_time = (self.generations - gen - 1) * generation_time
            if remaining_time > 60:
                remaining_time = '{0:.2f}m'.format(remaining_time / 60.0)
            else:
                remaining_time = '{0:.2f}s'.format(remaining_time)


            line_format = '{:>3} {:>10.2f} {:>10.2f} {:>12.4f} {:>12.2f} {:>12.2f} {:>12.3f} {:>12.4f} {:>12}'
            print(line_format.format(run_details['generation'][-1],
                                     run_details['average_length'][-1],
                                     run_details['average_depth'][-1],
                                     run_details['average_fitness'][-1],
                                     run_details['best_length'][-1],
                                     run_details['best_depth'][-1],
                                     run_details['best_fitness'][-1],
                                     run_details['best_oob_fitness'][-1],
                                     remaining_time))

    def fit_3D(self, X, y, baseline, sample_weight=None, need_parallel = True):
        """
        Fit the Genetic Program according to X, y.

        Parameters
        ----------
        X : array-like, shape = [n_dates, n_features, n_stocks]

        y : array-like, shape = [n_dates, n_stocks]

        baseline : float
            The baseline value to compare the fitness of the programs to => Factors better than baseline will be stored in self._satisfied_programs

        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples.

        need_parallel : bool, optional (default=True)
            Whether to use parallel processing when building the programs.

        Returns
        -------
        self : object
            Returns self.

        """
        random_state = check_random_state(self.random_state)

        hall_of_fame = self.hall_of_fame
        if hall_of_fame is None:
            hall_of_fame = self.population_size
        if hall_of_fame > self.population_size or hall_of_fame < 1:
            raise ValueError('hall_of_fame (%d) must be less than or equal to population_size (%d).' % (self.hall_of_fame, self.population_size))
        
        n_components = self.n_components
        if n_components is None:
            n_components = hall_of_fame
        if n_components > hall_of_fame or n_components < 1:
            raise ValueError('n_components (%d) must be less than or equal to hall_of_fame (%d).' % (self.n_components, self.hall_of_fame))

        self._function_set = []
        for function in self.function_set:
            if isinstance(function, str):
                if function not in _all_func_dictionary:
                    raise ValueError('invalid function name %s found in `function_set`.' % function)
                self._function_set.append(_all_func_dictionary[function])
            elif isinstance(function, _Function):
                self._function_set.append(function)
            else:
                raise ValueError('invalid type %s found in `function_set`.' % type(function))
        if not self._function_set:
            raise ValueError('No valid functions found in `function_set`.')

        # For point-mutation to find a compatible replacement node
        self._arities = {}
        for function in self._function_set:
            arity = function.arity
            self._arities[arity] = self._arities.get(arity, [])
            self._arities[arity].append(function)

        if isinstance(self.metric, _Fitness):
            self._metric = self.metric
        elif isinstance(self, TransformerMixin): 
            base_method = ('pearson', 'spearman')
            extra_method = tuple(_extra_map.keys())
            total_method = base_method + extra_method
            if self.metric not in total_method:
                raise ValueError('Unsupported metric: %s' % self.metric)
            self._metric = _fitness_map[self.metric]

        self._method_probs = np.array([self.p_crossover,
                                       self.p_subtree_mutation,
                                       self.p_hoist_mutation,
                                       self.p_point_mutation,
                                       self.p_reproduction])

        if self.init_method not in ('half and half', 'grow', 'full'):
            raise ValueError('Valid program initializations methods include "grow", "full" and "half and half". Given %s.' % self.init_method)

        if not ((isinstance(self.const_range, tuple) and len(self.const_range) == 2) or self.const_range is None):
            raise ValueError('const_range should be a tuple with length two, or None.')

        if (not isinstance(self.init_depth, tuple) or len(self.init_depth) != 2):
            raise ValueError('init_depth should be a tuple with length two.')
        if self.init_depth[0] > self.init_depth[1]:
            raise ValueError('init_depth should be in increasing numerical order: (min_depth, max_depth).')

        if self.transformer is not None:
            if isinstance(self.transformer, _Function):
                self._transformer = self.transformer
            elif self.transformer == 'sigmoid':
                self._transformer = sigmoid
            else:
                raise ValueError('Invalid `transformer`. Expected either "sigmoid" or _Function object, got %s' % type(self.transformer))
            if self._transformer.arity != 1:
                raise ValueError('Invalid arity for `transformer`. Expected 1, got %d.' % (self._transformer.arity))

        params = self.get_params()
        params['_metric'] = self._metric
        if hasattr(self, '_transformer'):
            params['_transformer'] = self._transformer
        else:
            params['_transformer'] = None
        params['function_set'] = self._function_set
        params['arities'] = self._arities
        params['method_probs'] = self._method_probs

        if not self.warm_start or not hasattr(self, '_programs'):
            # Free allocated memory, if any
            self._programs = []
            self.run_details_ = {'generation': [],
                                 'average_length': [],
                                 'average_depth': [],
                                 'average_fitness': [],
                                 'best_length': [],
                                 'best_depth': [],
                                 'best_fitness': [],
                                 'best_oob_fitness': [],
                                 'generation_time': []}

        prior_generations = len(self._programs)
        n_more_generations = self.generations - prior_generations

        if n_more_generations < 0:
            raise ValueError('generations=%d must be larger or equal to len(_programs)=%d when warm_start==True' % (self.generations, len(self._programs)))
        elif n_more_generations == 0:
            fitness = [program.raw_fitness_ for program in self._programs[-1]]
            warn('Warm-start fitting without increasing n_estimators does not fit new programs.')

        if self.warm_start:
            # Generate and discard seeds that would have been produced on the initial fit call.
            for i in range(len(self._programs)):
                _ = random_state.randint(MAX_INT, size=self.population_size)

        if self.verbose:
            # Print header fields.
            self._verbose_reporter()
            
        ##########################################GP Starts Here##########################################
        for gen in range(prior_generations, self.generations):
            start_time = time()

            if gen == 0:
                parents = None
            else:
                parents = self._programs[gen - 1]

            # Parallel loop
            n_jobs, n_programs, starts = _partition_estimators(self.population_size, self.n_jobs)
            seeds = random_state.randint(MAX_INT, size=self.population_size)
            if need_parallel:
                # Use loky as the backend for joblib
                population = Parallel(n_jobs=n_jobs, backend='loky', verbose=int(self.verbose > 1))(delayed(_parallel_evolve_3D)(n_programs[i],parents,X,y,sample_weight,seeds[starts[i]:starts[i + 1]],params) for i in range(n_jobs)) 
            else:
                population = []
                for i in range(n_jobs):
                    population.append(_parallel_evolve_3D(n_programs[i], parents, X, y, sample_weight, seeds[starts[i]:starts[i + 1]],params))

            # Reduce, maintaining order across different n_jobs
            population = list(itertools.chain.from_iterable(population))
            
            fitness = []
            length = []
            depth = []
            for program in population:
            # Find programs that are better than baseline and gather fitness and length
                if program.raw_fitness_ > baseline: 
                    program._satisfied = True
                    self._satisfied_programs.append(program)
                    last_day_df = pd.DataFrame(program.execute_3D(X)[-1],columns = [len(self._satisfied_programs) - 1])
                    if self._satisfied_factor_last_day_dfs is None:
                        self._satisfied_factor_last_day_dfs = last_day_df
                    else:
                        self._satisfied_factor_last_day_dfs = pd.concat([self._satisfied_factor_last_day_dfs, last_day_df], axis=1) 
                fitness.append(program.raw_fitness_)
                length.append(program.length_)
                depth.append(program.depth_)

            ##########################################Penalty Control Part##########################################
            # Penalize programs: 1. Highly correlated with satisfied programs 2. Large programs
            parsimony_coefficient = None
            if self.parsimony_coefficient == 'auto':
                parsimony_coefficient = (np.cov(length, fitness)[1, 0] / np.var(length))
            for program in population:
                program.fitness_ = program.fitness(parsimony_coefficient)
                
                last_day_series = pd.Series(program.execute_3D(X)[-1]) # Further reduce its reproductive opportunity if it is highly correlated with satisfied programs
                if getattr(program, '_satisfied', False): # If it is a satisfied program, exclude itself (corr = 1) from the correlation calculation
                    max_corr = self._satisfied_factor_last_day_dfs.corrwith(last_day_series,method='spearman').abs().nlargest(2).values[-1] 
                else:
                    max_corr = self._satisfied_factor_last_day_dfs.corrwith(last_day_series,method='spearman').abs().nlargest(1).values[-1]
                if max_corr > 0.8:
                    program.fitness_ = (0.6 + 1 - max_corr) * program.fitness_
            self._programs.append(population)

            # Remove old programs that didn't make it into the new population.
            if not self.low_memory:
                for old_gen in np.arange(gen, 0, -1):
                    indices = []
                    for program in self._programs[old_gen]:
                        if program is not None:
                            for idx in program.parents:
                                if 'idx' in idx:
                                    indices.append(program.parents[idx])
                    indices = set(indices)
                    for idx in range(self.population_size):
                        if idx not in indices:
                            self._programs[old_gen - 1][idx] = None
            elif gen > 0:
                # Remove old generations
                self._programs[gen - 1] = None

            # Record run details
            if self._metric.greater_is_better:
                best_program = population[np.argmax(fitness)]
            else:
                best_program = population[np.argmin(fitness)]

            self.run_details_['generation'].append(gen)
            self.run_details_['average_length'].append(np.mean(length))
            self.run_details_['average_depth'].append(np.mean(depth))
            self.run_details_['average_fitness'].append(np.mean(fitness))
            self.run_details_['best_length'].append(best_program.length_)
            self.run_details_['best_depth'].append(best_program.depth_)
            self.run_details_['best_fitness'].append(best_program.raw_fitness_)
            oob_fitness = np.nan
            if self.max_samples < 1.0:
                oob_fitness = best_program.oob_fitness_
            self.run_details_['best_oob_fitness'].append(oob_fitness)
            generation_time = time() - start_time
            self.run_details_['generation_time'].append(generation_time)

            if self.verbose:
                self._verbose_reporter(self.run_details_)

            # Check for early stopping
            if self._metric.greater_is_better:
                best_fitness = np.max(fitness)
                if best_fitness >= self.stopping_criteria:
                    break
            else:
                best_fitness = np.min(fitness)
                if best_fitness <= self.stopping_criteria:
                    break

        if isinstance(self, TransformerMixin):
            # Find the best 'hall_of_fame' individuals in the final generation
            fitness = np.array(fitness)
            if self._metric.greater_is_better:
                hall_of_fame = fitness.argsort()[::-1][:self.hall_of_fame]
            else:
                hall_of_fame = fitness.argsort()[:self.hall_of_fame]
            evaluation = np.array([gp.execute_3D(X).flatten() for gp in [self._programs[-1][i] for i in hall_of_fame]])

            with np.errstate(divide='ignore', invalid='ignore'):
                correlations = evaluation - np.nanmean(evaluation, axis=1).reshape((evaluation.shape[0], 1))
                correlations = np.abs(np.corrcoef(np.nan_to_num(correlations, nan=0.)))

            np.fill_diagonal(correlations, 0.)
            components = list(range(self.hall_of_fame))
            indices = list(range(self.hall_of_fame))

            # Iteratively remove least fit individual of most correlated pair
            while len(components) > self.n_components:
                most_correlated = np.unravel_index(np.argmax(correlations), correlations.shape)
                # The correlation matrix is sorted by fitness, so identifying
                # the least fit of the pair is simply getting the higher index
                worst = max(most_correlated)
                components.pop(worst)
                indices.remove(worst)
                correlations = correlations[:, indices][indices, :]
                indices = list(range(len(components)))
            self._best_programs = [self._programs[-1][i] for i in hall_of_fame[components]]
        else:
            # Find the best individual in the final generation
            if self._metric.greater_is_better:
                self._program = self._programs[-1][np.argmax(fitness)]
            else:
                self._program = self._programs[-1][np.argmin(fitness)]

class SymbolicTransformer(BaseSymbolic, TransformerMixin):
    """
    A Genetic Programming symbolic transformer.

    A symbolic transformer is a supervised transformer that begins by building
    a population of naive random formulas to represent a relationship. The
    formulas are represented as tree-like structures with mathematical
    functions being recursively applied to variables and constants. Each
    successive generation of programs is then evolved from the one that came
    before it by selecting the fittest individuals from the population to
    undergo genetic operations such as crossover, mutation or reproduction.
    The final population is searched for the fittest individuals with the least
    correlation to one another.

    Parameters
    ----------
    population_size : integer, optional (default=1000)
        The number of programs in each generation.

    hall_of_fame : integer, or None, optional (default=100)
        The number of fittest programs to compare from when finding the
        least-correlated individuals for the n_components. If `None`, the
        entire final generation will be used.

    n_components : integer, or None, optional (default=10)
        The number of best programs to return after searching the hall_of_fame
        for the least-correlated individuals. If `None`, the entire
        hall_of_fame will be used.

    generations : integer, optional (default=20)
        The number of generations to evolve.

    tournament_size : integer, optional (default=20)
        The number of programs that will compete to become part of the next
        generation.

    stopping_criteria : float, optional (default=1.0)
        The required metric value required in order to stop evolution early.

    const_range : tuple of two floats, or None, optional (default=(-1., 1.))
        The range of constants to include in the formulas. If None then no
        constants will be included in the candidate programs.

    init_depth : tuple of two ints, optional (default=(2, 6))
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.
        When combined with `init_method='half and half'` this yields the well-
        known 'ramped half and half' initialization method.

    init_method : str, optional (default='half and half')
        - 'grow' : Nodes are chosen at random from both functions and
          terminals, allowing for smaller trees than `init_depth` allows. Tends
          to grow asymmetrical trees.
        - 'full' : Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
        - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
          'grow', making for a mix of tree shapes in the initial population.

    function_set : iterable, optional (default=('add', 'sub', 'mul', 'div'))
        The functions to use when building and evolving programs. This iterable
        can include strings to indicate either individual functions as outlined
        below, or you can also include your own functions as built using the
        ``make_function`` factory from the ``functions`` module.

    metric : str, optional (default='pearson')
        The name of the raw fitness metric. Available options include:

        - 'pearson', for Pearson's product-moment correlation coefficient.
        - 'spearman' for Spearman's rank-order correlation coefficient.

    parsimony_coefficient : float or "auto", optional (default=0.001)
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. Bloat is when
        evolution is increasing the size of programs without a significant
        increase in fitness, which is costly for computation time and makes for
        a less understandable final result. This parameter may need to be tuned
        over successive runs.

        If "auto" the parsimony coefficient is recalculated for each generation
        using c = Cov(l,f)/Var( l), where Cov(l,f) is the covariance between
        program size l and program fitness f in the population, and Var(l) is
        the variance of program sizes.

    p_crossover : float, optional (default=0.9)
        The probability of performing crossover on a tournament winner.
        Crossover takes the winner of a tournament and selects a random subtree
        from it to be replaced. A second tournament is performed to find a
        donor. The donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring in the next
        generation.

    p_subtree_mutation : float, optional (default=0.01)
        The probability of performing subtree mutation on a tournament winner.
        Subtree mutation takes the winner of a tournament and selects a random
        subtree from it to be replaced. A donor subtree is generated at random
        and this is inserted into the original parent to form an offspring in
        the next generation.

    p_hoist_mutation : float, optional (default=0.01)
        The probability of performing hoist mutation on a tournament winner.
        Hoist mutation takes the winner of a tournament and selects a random
        subtree from it. A random subtree of that subtree is then selected
        and this is 'hoisted' into the original subtrees location to form an
        offspring in the next generation. This method helps to control bloat.

    p_point_mutation : float, optional (default=0.01)
        The probability of performing point mutation on a tournament winner.
        Point mutation takes the winner of a tournament and selects random
        nodes from it to be replaced. Terminals are replaced by other terminals
        and functions are replaced by other functions that require the same
        number of arguments as the original node. The resulting tree forms an
        offspring in the next generation.

        Note : The above genetic operation probabilities must sum to less than
        one. The balance of probability is assigned to 'reproduction', where a
        tournament winner is cloned and enters the next generation unmodified.

    p_point_replace : float, optional (default=0.05)
        For point mutation only, the probability that any given node will be
        mutated.
    
    p_reproduction : float, optional (default=0.01)
        The probability of performing reproduction on a tournament winner.

    max_samples : float, optional (default=1.0)
        The fraction of samples to draw from X to evaluate each program on.

    feature_names : list, optional (default=None)
        Optional list of feature names, used purely for representations in
        the `print` operation or `export_graphviz`. If None, then X0, X1, etc
        will be used for representations.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more generations to the evolution, otherwise, just fit a new
        evolution.

    low_memory : bool, optional (default=False)
        When set to ``True``, only the current generation is retained. Parent
        information is discarded. For very large populations or runs with many
        generations, this can result in substantial memory use reduction.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for `fit`. If -1, then the number
        of jobs is set to the number of cores.

    verbose : int, optional (default=0)
        Controls the verbosity of the evolution building process.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    run_details_ : dict
        Details of the evolution process. Includes the following elements:

        - 'generation' : The generation index.
        - 'average_length' : The average program length of the generation.
        - 'average_depth' : The average program depth of the generation.
        - 'average_fitness' : The average program fitness of the generation.
        - 'best_length' : The length of the best program in the generation.
        - 'best_depth' : The depth of the best program in the generation.
        - 'best_fitness' : The fitness of the best program in the generation.
        - 'best_oob_fitness' : The out of bag fitness of the best program in
          the generation (requires `max_samples` < 1.0).
        - 'generation_time' : The time it took for the generation to evolve.

    References
    ----------
    .. [1] J. Koza, "Genetic Programming", 1992.

    .. [2] R. Poli, et al. "A Field Guide to Genetic Programming", 2008.

    """

    def __init__(self,
                 *,
                 population_size=1000,
                 hall_of_fame=100,
                 n_components=10,
                 generations=20,
                 tournament_size=20,
                 stopping_criteria=1.0,
                 const_range=(-1., 1.),
                 init_depth=(2, 6),
                 init_method='half and half',
                 function_set=('add', 'sub', 'mul', 'div'),
                 metric='pearson',
                 parsimony_coefficient=0.001,
                 p_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_point_replace=0.05,
                 p_reproduction = 0.01,
                 max_samples=1.0,
                 feature_names=None,
                 warm_start=False,
                 low_memory=False,
                 n_jobs=1,
                 verbose=0,
                 random_state=None):
        
        super(SymbolicTransformer, self).__init__(
            population_size=population_size,
            hall_of_fame=hall_of_fame,
            n_components=n_components,
            generations=generations,
            tournament_size=tournament_size,
            stopping_criteria=stopping_criteria,
            const_range=const_range,
            init_depth=init_depth,
            init_method=init_method,
            function_set=function_set,
            metric=metric,
            parsimony_coefficient=parsimony_coefficient,
            p_crossover=p_crossover,
            p_subtree_mutation=p_subtree_mutation,
            p_hoist_mutation=p_hoist_mutation,
            p_point_mutation=p_point_mutation,
            p_point_replace=p_point_replace,
            p_reproduction=p_reproduction,
            max_samples=max_samples,
            feature_names=feature_names,
            warm_start=warm_start,
            low_memory=low_memory,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state)

    def __str__(self):
        """Overloads `print` output of the object to resemble LISP trees."""
        if not hasattr(self, '_best_programs'):
            return self.__repr__()
        output = str([gp.__str__() for gp in self._best_programs])
        return output.replace("',", ",\n").replace("'", "")
    
    def show_program(self, baseline=False):
        """
        Generates a BRIEF report for each program in the genetic algorithm.

        Parameters:
        ----------
            baseline (bool): If True, programs that are better than the baseline value will all be included in the report. Otherwise, only the last round survived factors will be included.
        
        Returns:
        ----------
            list: A list of dictionaries, where each dictionary contains the following information:
                - "Expression": The string representation of the program.
                - "Fitness": The raw fitness value of the program.
                - "OOB Fitness": The out-of-bag fitness value of the program.
        
        """

        result = []
        
        for program in self._satisfied_programs if baseline else self._best_programs:
                result.append({"Expression": program.__str__(), "Fitness": program.raw_fitness_,  "OOB Fitness": program.oob_fitness_, 'Length': program.length_, 'Depth': program.depth_})
        return result
