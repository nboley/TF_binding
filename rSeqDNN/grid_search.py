import ast
import types
from moe.easy_interface.experiment import Experiment
from moe.easy_interface.simple_endpoint import gp_next_points
from moe.optimal_learning.python.data_containers import SamplePoint

def is_valid_python(code):
   try:
       ast.parse(code)
   except SyntaxError:
       return False
   return True

def find_method(obj, method):
    try:
        if isinstance(method, str):
            return getattr(obj, method)
        else:
            return types.MethodType(method, obj)
    except AttributeError as e:
        print method, 'method not found for object ', obj
        raise e

class MOESearch(object):
    def __init__(self, estimator, param_grid, fixed_param=None, conditional_param=None):
        """
        MOE hyper parameter search. Limited to discrete parameters only.
        
        Parameters
        ----------
        estimator : estimator classobj
            estimator needs to provide fit and score methods.
        param_grid : dict
            Dictionary with parameter names as keys and lists of
            lower and upper bounds to search over as values.
        fixed_param : dict, optional
            Parameters that remain fixed during the search.
        conditional_param : dict, optional
            Parameters conditional on the param_grid.
            For example, {maxpool_stride: 'maxpool_size/2'}.
        """ 
        self.estimator = estimator
        assert all(len(value)==2 and value[1]>value[0] for key, value 
                   in param_grid.iteritems()), \
        "Invalid parameter grid!"
        self.param_grid = param_grid
        self.fixed_param = fixed_param if fixed_param is not None else {}
        self.conditional_param = conditional_param if conditional_param is not None else {}
        # start MOE experiment
        self.experiment = Experiment(self.param_grid.values())

    def fit(self, fit_method, fit_param, score_method, score_param, max_iter=2):
        """
        Runs MOE search over the parameter grid.
        
        Parameters
        ----------
        fit_method : string or callable
        fit_param : dict
            Dictionary with parameter names as keys and
            parameter values as values to pass to fit_method
        score_method : string or callable
        score_param : dict
            Dictionary with parameter names as keys and
            parameter values as values to pass to score_method
        max_iter : int, default=2
            Maximum number of search iterations.

        Returns
        -------
        self

        Attributes
        ----------
        self.experiment : MOE experiment
        """
        param_names = self.param_grid.keys()
        for i in xrange(max_iter):
            estimator_param = self.fixed_param.copy()
            # sample next search iteration with MOE
            next_point_to_sample = [int(round(i)) for i in gp_next_points(self.experiment)[0]]
            search_param = dict(zip(param_names, next_point_to_sample))
            # add sampled parameters
            estimator_param.update(search_param)
            # evaluate conditional parameters and add
            curr_conditional_param = self.conditional_param.copy()
            for key, value in curr_conditional_param.iteritems():
                for name in param_names:
                    if name in value:
                        value = value.replace(name, str(search_param[name]))
                curr_conditional_param[key] = eval(value)
            estimator_param.update(curr_conditional_param)
            # initialize estimators with new parameters
            model = self.estimator(**estimator_param)
            # fit, score, store results
            print fit_method
            fit = find_method(model, fit_method)
            print fit
            fit_model = fit(**fit_param)
            score = find_method(fit_model, score_method)
            value_of_next_point = score(**score_param)
            print 'value of next point: ', value_of_next_point
            self.experiment.historical_data.append_sample_points(
                [SamplePoint(next_point_to_sample, value_of_next_point)])

        return self
